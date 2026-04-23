#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binding energy (interface mode) follows D090 calc_binding_energy:
  - Locally repack interface residues within <pack_radius> of the position
  - Score complex (before separation)
  - Translate partner group 2 by 500 Å
  - Locally repack again (unbound)
  - ΔG_bind = score_bound - score_unbound

PDB output:
  --save_pdbs none | all | select
  --save_pdb_select  e.g. A42G,A100W   (chain + pdb_resnum + mutAA)
  --save_wt_pdb true | false           (also dump the per-position processed WT)

Modes (controlled by --partners):
  1) Interface / binding mode:  --partners A_B  (underscore present)
  2) Single-body mode:          --partners A    (no underscore)

Output columns (interface mode):
  wt_bind_energy / mut_bind_energy   — ΔG_bind (complex − separated), D090 protocol
  wt_total_energy / mut_total_energy — E_total of full complex (no separation)
  ddg_bind  — ΔΔG_binding = mut_bind_energy  − wt_bind_energy
  ddg_fold  — ΔG_folding  = mut_total_energy − wt_total_energy
"""

import os
import csv
import sys
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import pyrosetta
from pyrosetta import Pose, create_score_function
from pyrosetta.rosetta import core, protocols
from pyrosetta.rosetta.utility import vector1_bool
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.select.residue_selector import (
    ChainSelector,
    NeighborhoodResidueSelector,
    OrResidueSelector,
)
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.pack.task.operation import (
    RestrictToRepacking,
    PreventRepacking,
)
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue


# Canonical amino acids
CANONICAL_ONE = list("ACDEFGHIKLMNPQRSTVWY")

# ncAA helpers
def collect_params_files(extra_res_path: str) -> list:
    folder = Path(extra_res_path)
    if not folder.is_dir():
        sys.exit(f"[error] --extra_res_path '{extra_res_path}' is not a directory.")
    params = sorted(folder.glob("*.params"))
    if not params:
        sys.stderr.write(
            f"[warn] --extra_res_path '{extra_res_path}' contains no *.params files.\n"
        )
    return [str(p) for p in params]


def is_ncaa(aa_token: str) -> bool:
    """
    Return True if token looks like an ncAA 3-letter code rather than a
    canonical 1-letter code.  We treat any token with length != 1 as ncAA.
    """
    return len(aa_token) != 1


def validate_ncaa_residue_type(pose, ncaa_name: str) -> bool:
    """
    Check that the ncAA residue type name is present in the pose's
    ResidueTypeSet 
    """
    rts = pose.residue_type_set_for_pose()
    return rts.has_name(ncaa_name)


# Helpers
def safe_flush(fh):
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except Exception:
        pass


def is_interface_mode(partners: str) -> bool:
    return "_" in (partners or "")


def normalize_chain_arg(chain_arg: str):
    s = (chain_arg or "").strip().replace(" ", "").replace(",", "")
    return set(list(s)) if s else set()


def partner_chain_sets(partners: str):
    if "_" not in partners:
        raise ValueError("--partners must look like A_B or AB_C in interface mode")
    a, b = partners.split("_", 1)
    return set(a.replace(",", "")), set(b.replace(",", ""))


def parse_mutant_targets(mutant_aa: str):
    """
    Parse --mutant_aa into a list of mutation targets.

    Canonical AAs: 1-letter codes (A, G, V, …).
    Non-canonical AAs: Rosetta 3-letter residue type names as written in the
                       corresponding .params file (e.g. NLU, AIB, B3A).
    Tokens are comma-separated and may be mixed, e.g.:  A,NLU,AIB,V

    'all' expands to the 20 canonical amino acids only; ncAAs are never
    included automatically.
    """
    if mutant_aa is None:
        raise ValueError("--mutant_aa is required")
    if mutant_aa.strip().lower() == "all":
        return CANONICAL_ONE[:]
    targets = []
    seen = set()
    for tok in mutant_aa.split(","):
        aa = tok.strip()
        if not aa:
            continue
        if is_ncaa(aa):
            # ncAA: store the 3-letter name as-is (case preserved from params)
            key = aa  # do NOT force upper — Rosetta type names are case-sensitive
            if key not in seen:
                targets.append(key)
                seen.add(key)
        else:
            # Canonical: 1-letter, force upper and validate
            aa_up = aa.upper()
            if aa_up not in CANONICAL_ONE:
                raise ValueError(f"Invalid canonical amino acid '{aa_up}' in --mutant_aa. "
                                 f"For non-canonical residues supply the Rosetta 3-letter name "
                                 f"(e.g. NLU, AIB) as found in the .params file.")
            if aa_up not in seen:
                targets.append(aa_up)
                seen.add(aa_up)
    if not targets:
        raise ValueError("--mutant_aa produced no valid mutation targets")
    return targets

def select_interface_residues(pose, partners, cutoff):
    chains1, chains2 = partner_chain_sets(partners)
    sel1 = ChainSelector("".join(sorted(chains1)))
    sel2 = ChainSelector("".join(sorted(chains2)))
    neigh1 = NeighborhoodResidueSelector(sel2, cutoff, True)
    neigh2 = NeighborhoodResidueSelector(sel1, cutoff, True)
    iface = OrResidueSelector(neigh1, neigh2)
    subset = iface.apply(pose)
    return sorted(get_residues_from_subset(subset))

def pose_res_label(pose, resi: int):
    pdbi = pose.pdb_info()
    return pdbi.chain(resi), pdbi.number(resi)

def partner_chain2_start(pose, partners):
    """Return the first pose index belonging to partner group 2."""
    _, chains2 = partner_chain_sets(partners)
    pdbi = pose.pdb_info()
    for i in range(1, pose.total_residue() + 1):
        if pdbi.chain(i) in chains2:
            return i
    raise RuntimeError("Could not find chain2 start residue.")

# TaskFactory helpers
def _make_local_repack_task(pose, center_resi: int, radius: float):
    """
    Build a TaskFactory that restricts to repacking within <radius> of center_resi.
    """
    tf = core.pack.task.TaskFactory()
    tf.push_back(RestrictToRepacking())
    prevent = PreventRepacking()
    center = pose.residue(center_resi).nbr_atom_xyz()
    for i in range(1, pose.total_residue() + 1):
        if center.distance_squared(pose.residue(i).nbr_atom_xyz()) > radius ** 2:
            prevent.include_residue(i)
    tf.push_back(prevent)
    return tf

def _make_packer_task(pose):
    """
    Robust PackerTask constructor across PyRosetta versions.
    """
    try:
        return core.pack.task.standard_packer_task(pose)
    except AttributeError:
        TF = core.pack.task.TaskFactory()
        from pyrosetta.rosetta.core.pack.task.operation import InitializeFromCommandline
        TF.push_back(InitializeFromCommandline())
        return TF.create_task_and_apply_taskoperations(pose)


def _make_mutation_task(pose, mutant_position: int, aa: str, pack_radius: float):
    """
    Build a PackerTask that forces position mutant_position to aa, repacking
    residues within pack_radius. All other positions are frozen.

    Only used for canonical (1-letter) amino acids.  Non-canonical residues
    are handled separately via MutateResidue + _make_local_repack_task.
    """
    task = _make_packer_task(pose)
    aa_bool = vector1_bool()
    aa_enum = core.chemical.aa_from_oneletter_code(aa)
    for i in range(1, 21):
        aa_bool.append(i == int(aa_enum))
    task.nonconst_residue_task(mutant_position).restrict_absent_canonical_aas(aa_bool)
    center = pose.residue(mutant_position).nbr_atom_xyz()
    for i in range(1, pose.total_residue() + 1):
        if i != mutant_position and center.distance_squared(
            pose.residue(i).nbr_atom_xyz()
        ) > pack_radius ** 2:
            task.nonconst_residue_task(i).prevent_repacking()
    return task


# FastRelax helpers
def _movemap_chain(pose, mutate_chains: set):
    pdbi = pose.pdb_info()
    mm = pyrosetta.MoveMap()
    for i in range(1, pose.total_residue() + 1):
        on = pdbi.chain(i) in mutate_chains
        mm.set_bb(i, on)
        mm.set_chi(i, on)
    return mm

def _movemap_local(pose, center_resi: int, radius: float):
    mm = pyrosetta.MoveMap()
    c = pose.residue(center_resi).nbr_atom_xyz()
    r2 = radius * radius
    for i in range(1, pose.total_residue() + 1):
        on = c.distance_squared(pose.residue(i).nbr_atom_xyz()) <= r2
        mm.set_bb(i, on)
        mm.set_chi(i, on)
    return mm

def _apply_fastrelax(pose, mm, scorefxn, cycles: int):
    fr = FastRelax(int(cycles))
    fr.set_scorefxn(scorefxn)
    fr.set_movemap(mm)
    fr.apply(pose)


# Core ΔΔG protocol
def calc_binding_energy_d090(pose, scorefxn, center_resi: int, cutoff: float, partners: str):
    """
    D090 calc_binding_energy protocol.
    The pose passed in is already repacked/relaxed — do NOT repack before scoring complex.
    1. Clone pose.
    2. Score complex as-is (bound state, already processed).
    3. Translate partner group 2 by 500 Å.
    4. Repack unbound (each chain relaxes into its free conformation).
    5. Return score_bound - score_unbound.
    """
    test_pose = pose.clone()

    # Score the bound complex directly — pose is already in its processed state
    before = float(scorefxn(test_pose))

    # Translate partner group 2 by 500 Å
    _, chains2 = partner_chain_sets(partners)
    xyz = xyzVector_double_t()
    xyz.x = 500.0
    xyz.y = 0.0
    xyz.z = 0.0
    pdbi = test_pose.pdb_info()
    for r in range(1, test_pose.total_residue() + 1):
        if pdbi.chain(r) in chains2:
            for a in range(1, test_pose.residue(r).natoms() + 1):
                test_pose.residue(r).set_xyz(a, test_pose.residue(r).xyz(a) + xyz)

    # Repack unbound — let each chain settle into its free conformation
    tf = _make_local_repack_task(test_pose, center_resi, cutoff)
    packer = protocols.minimization_packing.PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(test_pose)

    after = float(scorefxn(test_pose))
    return before - after


def make_global_wt(pose, scorefxn, use_fastrelax: bool, relax_scope: str,
                   relax_radius: float, relax_cycles: int, mutate_chains: set):
    """
    Relax the full WT pose ONCE globally before any scanning begins.
    This single relaxed structure is the fixed reference for all ΔΔG calculations —
    WT folding and binding energies will be identical across all residues and trials.
    """
    wt = pose.clone()
    if use_fastrelax:
        if relax_scope == "chain":
            mm = _movemap_chain(wt, mutate_chains)
        else:
            # fall back to full backbone if no center residue available globally
            mm = pyrosetta.MoveMap()
            mm.set_bb(True)
            mm.set_chi(True)
        _apply_fastrelax(wt, mm, scorefxn, relax_cycles)
    return wt


def make_mutant(wt_processed: Pose, resi: int, aa: str, pack_radius: float,
                scorefxn, use_fastrelax: bool, relax_scope: str, relax_radius: float,
                relax_cycles: int, mutate_chains: set):
    """
    Clone the processed WT, mutate position resi to aa, repack + optional relax.
    Starting from wt_processed ensures WT and mutant share the same backbone state.

    For canonical AAs (1-letter code): uses PackRotamersMover with a restricted
    PackerTask (restrict_absent_canonical_aas), as in v5.

    For non-canonical AAs (Rosetta 3-letter type name): uses MutateResidue to
    place the ncAA rotamer, then runs a local PackRotamersMover to repack
    neighbouring residues within pack_radius.
    """
    mut = wt_processed.clone()

    if is_ncaa(aa):
        # --- ncAA path ---
        # MutateResidue places the best rotamer for the target residue type.
        mutator = MutateResidue(resi, aa)
        mutator.apply(mut)
        # Repack neighbours (excluding the newly mutated position to preserve
        # the placed rotamer; set restrict_to_repacking on neighbours only).
        tf = _make_local_repack_task(mut, resi, pack_radius)
        packer = protocols.minimization_packing.PackRotamersMover(scorefxn)
        packer.task_factory(tf)
        packer.apply(mut)
    else:
        # --- Canonical AA path ---
        task = _make_mutation_task(mut, resi, aa, pack_radius)
        protocols.minimization_packing.PackRotamersMover(scorefxn, task).apply(mut)

    if use_fastrelax:
        if relax_scope == "chain":
            mm = _movemap_chain(mut, mutate_chains)
        else:
            mm = _movemap_local(mut, resi, relax_radius)
        _apply_fastrelax(mut, mm, scorefxn, relax_cycles)

    return mut


def score_binding(pose, scorefxn, resi: int, pack_radius: float,
                  partners: str, mode: str) -> float:
    """
    ΔG_bind — always computed, regardless of mode.

    Interface mode: full D090 calc_binding_energy_d090 protocol (separate partner chains).
    Single mode:    same D090 separation logic but translates the second half of the
                    pose (residues after the first chain boundary) as a proxy partner.
                    For a genuine single chain this collapses to total-energy ΔΔG, but
                    keeps the scoring pathway identical so output columns are consistent.
    """
    if mode == "interface":
        return calc_binding_energy_d090(pose, scorefxn, resi, pack_radius, partners)
    else:
        return _calc_binding_energy_single(pose, scorefxn, resi, pack_radius)


def _calc_binding_energy_single(pose, scorefxn, center_resi: int, cutoff: float) -> float:
    """
    D090-style binding energy for single-chain (or non-interface) mode.

    Mirrors calc_binding_energy_d090 exactly, except that the "partner 2"
    is defined as all residues after the first chain boundary (i.e., residues
    belonging to any chain other than the chain of residue 1).  If the pose
    truly has only one chain, ALL residues belong to one chain so nothing is
    translated and ΔG_bind = 0 (score_bound − score_unbound = 0).  This is
    correct — there is no separable partner — and ΔΔG_bind will then equal
    ΔΔG_fold, which is the right result for single-chain mutations.
    """
    test_pose = pose.clone()

    before = float(scorefxn(test_pose))

    # Identify "partner 2": all chains other than the chain of residue 1
    pdbi = test_pose.pdb_info()
    chain1 = pdbi.chain(1)
    partner2_residues = [
        r for r in range(1, test_pose.total_residue() + 1)
        if pdbi.chain(r) != chain1
    ]

    if partner2_residues:
        xyz = xyzVector_double_t()
        xyz.x = 500.0
        xyz.y = 0.0
        xyz.z = 0.0
        for r in partner2_residues:
            for a in range(1, test_pose.residue(r).natoms() + 1):
                test_pose.residue(r).set_xyz(a, test_pose.residue(r).xyz(a) + xyz)

    tf = _make_local_repack_task(test_pose, center_resi, cutoff)
    packer = protocols.minimization_packing.PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(test_pose)

    after = float(scorefxn(test_pose))
    return before - after


def score_total(pose, scorefxn) -> float:
    """
    E_total of the full complex with no separation. Used for ddg_fold.
    """
    return float(scorefxn(pose))


def get_per_term_energies(pose, scorefxn) -> dict:
    """
    Return a dict mapping score term name -> weighted total energy for all
    active (non-zero weight) terms in scorefxn, following the D090 convention
    of scoring the pose before extracting energies.

    Sums residue_total_energies across all residues and multiplies by the
    ScoreFunction weight, giving the same values as seen in a score file.
    """
    # Ensure energies are up to date
    scorefxn(pose)
    energies = pose.energies()
    weights = scorefxn.weights()

    # Iterate over all ScoreTypes; collect active (weight != 0) terms
    term_scores = {}
    for st in core.scoring.ScoreType.__members__.values():
        w = weights[st]
        if w == 0.0:
            continue
        name = core.scoring.name_from_score_type(st)
        total = 0.0
        for r in range(1, pose.total_residue() + 1):
            rte = energies.residue_total_energies(r)
            total += float(rte[st])
        term_scores[name] = round(w * total, 4)

    return term_scores


# Main
def main():
    import argparse

    ap = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        description="ΔΔG scanning — correct per-position WT baseline + PDB output",
    )

    def convert_arg_line_to_args(line):
        line = line.strip()
        if not line or line.startswith("#"):
            return []
        return line.split()
    ap.convert_arg_line_to_args = convert_arg_line_to_args

    # Core
    ap.add_argument("--pdb_filename", required=True)
    ap.add_argument("--partners", required=True, help="Interface: A_B  | Single: A")
    ap.add_argument("--mutant_aa", required=True,
                    help="all | A | A,V,Y,F,W | A,NLU,AIB  "
                         "(canonical: 1-letter codes; ncAA: Rosetta 3-letter type name from .params)")
    ap.add_argument("--extra_res_path", default="",
                    help="Path to a folder of Rosetta .params files for non-canonical amino acids. "
                         "All *.params files in the folder are loaded at init via -extra_res_fa. "
                         "Required when --mutant_aa contains any ncAA 3-letter type names.")
    ap.add_argument("--mutate_chain", default="", help="Chains to mutate (e.g. A or A,C). Default: left partner group.")
    ap.add_argument("--interface_cutoff", type=float, default=6.0, help="Interface detection cutoff (interface mode only).")
    ap.add_argument("--pack_radius", type=float, default=8.0, help="Repack radius (Å) around mutated residue.")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--scorefxn", default="ref2015")
    ap.add_argument("--use_fastrelax", default="false")
    ap.add_argument("--relax_scope", default="chain", choices=["chain", "local"])
    ap.add_argument("--relax_radius", type=float, default=8.0, help="Used only when --relax_scope local.")
    ap.add_argument("--relax_cycles", type=int, default=3)
    ap.add_argument("--trial_output", default="ddg_output")

    # PDB output
    ap.add_argument("--save_pdbs", default="none", choices=["none", "all", "select"],
                    help="Save mutant PDBs: none | all | select (pair with --save_pdb_select).")
    ap.add_argument("--save_pdb_select", default="",
                    help="Comma-separated mutation tags to save, e.g. A42G,A100W (chain+pdb_resnum+mutAA).")
    ap.add_argument("--save_wt_pdb", default="false",
                    help="If true, also save the processed WT PDB for each saved mutant position.")
    ap.add_argument("--scan_residues", default="",
                    help="Restrict scan to specific residues, e.g. A42,A100,B15 (chain+pdb_resnum). "
                         "Default: scan all eligible residues.")
    ap.add_argument("--force_scan_residues", default="false",
                    help="If true, residues in --scan_residues bypass the interface/chain candidate "
                         "filter and are included directly (must still be protein residues on the pose). "
                         "Useful for residues just outside the interface cutoff.")

    args = ap.parse_args()

    # --- Build PyRosetta init flags ---
    init_flags = ["-ignore_unrecognized_res false"]

    if args.extra_res_path.strip():
        params_files = collect_params_files(args.extra_res_path.strip())
        if params_files:
            # -extra_res_fa accepts a space-separated list of .params paths
            init_flags.append("-extra_res_fa " + " ".join(params_files))
            sys.stderr.write(
                f"[info] Loading {len(params_files)} ncAA params file(s) "
                f"from '{args.extra_res_path}':\n"
            )
            for p in params_files:
                sys.stderr.write(f"       {p}\n")

    pyrosetta.init(" ".join(init_flags))

    # Load pose
    pose = Pose()
    core.import_pose.pose_from_file(pose, args.pdb_filename)
    if not pose.is_fullatom():
        sys.stderr.write("[warn] Pose is not full-atom. ΔΔG results may be meaningless.\n")

    pdbi = pose.pdb_info()
    mode = "interface" if is_interface_mode(args.partners) else "single"
    scorefxn = create_score_function(args.scorefxn)
    targets = parse_mutant_targets(args.mutant_aa)

    # Validate any ncAA targets against the loaded ResidueTypeSet
    for aa in targets:
        if is_ncaa(aa):
            if not validate_ncaa_residue_type(pose, aa):
                sys.exit(
                    f"[error] ncAA residue type '{aa}' was not found in the loaded "
                    f"ResidueTypeSet. Check that its .params file is present in "
                    f"--extra_res_path and that the name in --mutant_aa matches the "
                    f"'NAME' field of the .params file exactly (case-sensitive)."
                )

    # Resolve mutate_chains
    if mode == "interface":
        left_chains, _ = partner_chain_sets(args.partners)
        default_mut_chains = set(left_chains)
    else:
        default_mut_chains = normalize_chain_arg(args.partners)
        if not default_mut_chains:
            sys.exit("[error] Single mode requires --partners like A (or AC).")
    mutate_chains = normalize_chain_arg(args.mutate_chain) if args.mutate_chain else default_mut_chains

    # Candidate residues
    if mode == "interface":
        candidates = select_interface_residues(pose, args.partners, float(args.interface_cutoff))
    else:
        candidates = [i for i in range(1, pose.total_residue() + 1) if pdbi.chain(i) in mutate_chains]

    scan_res = [
        i for i in candidates
        if (pdbi.chain(i) in mutate_chains) and pose.residue(i).is_protein()
    ]

    # Optional: filter to specific residues via --scan_residues A42,A100,B15
    if args.scan_residues.strip():
        force = str(args.force_scan_residues).strip().lower() == "true"
        wanted = set()
        for tok in args.scan_residues.split(","):
            tok = tok.strip().upper()
            if not tok:
                continue
            chain_part = tok[0]
            try:
                resnum_part = int(tok[1:])
            except ValueError:
                sys.exit(f"[error] Could not parse --scan_residues token '{tok}'. "
                         f"Expected format: chain+resnum e.g. A42")
            wanted.add((chain_part, resnum_part))

        if force:
            # Bypass interface/chain candidate filter entirely.
            # scan_res becomes ONLY the explicitly requested residues.
            scan_res = [
                i for i in range(1, pose.total_residue() + 1)
                if pose.residue(i).is_protein()
                and (pdbi.chain(i), pdbi.number(i)) in wanted
            ]
        else:
            scan_res = [
                i for i in scan_res
                if (pdbi.chain(i), pdbi.number(i)) in wanted
            ]

        # Warn about any requested residues still not found
        found = {(pdbi.chain(i), pdbi.number(i)) for i in scan_res}
        for chain_part, resnum_part in sorted(wanted):
            if (chain_part, resnum_part) not in found:
                sys.stderr.write(
                    f"[warn] --scan_residues: {chain_part}{resnum_part} not found "
                    f"({'forced lookup' if force else 'interface candidates'}) — "
                    f"check chain ID and residue number.\n"
                )

    if not scan_res:
        sys.stderr.write(
            f"[warn] No residues selected. mode={mode} partners={args.partners} "
            f"mutate_chains={''.join(sorted(mutate_chains))}\n"
        )

    # PDB output setup
    use_fastrelax = str(args.use_fastrelax).strip().lower() == "true"
    relax_scope = str(args.relax_scope).strip().lower()
    save_pdbs = args.save_pdbs
    save_wt_pdb = str(args.save_wt_pdb).strip().lower() == "true"
    select_set = {s.strip() for s in args.save_pdb_select.split(",") if s.strip()} \
        if save_pdbs == "select" else set()
    # Normalise canonical 1-letter entries to upper; leave ncAA names as-is
    select_set = {
        s.upper() if len(s) <= 4 and s[-1:].isupper() else s
        for s in select_set
    }
    pdb_dir = Path(str(args.trial_output) + "_pdbs") if save_pdbs != "none" else None

    def should_save(chain, pdb_resnum, aa):
        if save_pdbs == "all":
            return True
        if save_pdbs == "select":
            return f"{chain}{pdb_resnum}{aa}" in select_set
        return False

    # Streaming CSV output
    # Targeted runs (--scan_residues) always overwrite so they don't append to a prior full scan.
    # Full scans append so they can be resumed after interruption.
    out_stream = Path(f"{args.trial_output}.csv")
    targeted = bool(args.scan_residues.strip())
    file_mode = "w" if targeted else "a"
    newfile = targeted or not out_stream.exists()
    ddg_trials = defaultdict(list)

    # -------------------------------------------------------------------
    # Build ONE global relaxed WT before scanning.
    # WT folding and binding energies are fixed — identical across all
    # residues, AAs, and trials. Only mutants are sampled stochastically.
    # -------------------------------------------------------------------
    sys.stderr.write("[info] Building global relaxed WT reference...\n")
    global_wt = make_global_wt(
        pose, scorefxn,
        use_fastrelax, relax_scope, float(args.relax_radius),
        int(args.relax_cycles), mutate_chains,
    )

    # Discover active score terms from a single scoring of the WT.
    # Term order is fixed once here so all rows have identical columns.
    _probe_terms = get_per_term_energies(global_wt, scorefxn)
    active_terms = list(_probe_terms.keys())

    with out_stream.open(file_mode, newline="") as fh:
        w = csv.writer(fh)

        if newfile:
            # Unified schema — binding ddG always present regardless of mode
            per_term_headers = (
                [f"wt_{term_name}" for term_name in active_terms]
                + [f"mut_{term_name}" for term_name in active_terms]
                + [f"delta_{term_name}" for term_name in active_terms]
            )
            w.writerow([
                "pose_index", "chain", "residue_number",
                "wt_aa", "mut_aa", "trial",
                # Binding energy (D090 separation protocol — always computed)
                "wt_bind_energy", "mut_bind_energy", "ddg_bind",
                "avg_ddg_bind", "sd_ddg_bind",
                # Total / folding energy
                "wt_total_energy", "mut_total_energy", "ddg_fold",
                "avg_ddg_fold", "sd_ddg_fold",
                "mode", "relax_scope", "mutate_chains",
            ] + per_term_headers)
            safe_flush(fh)

        for resi in scan_res:
            wt_aa = pose.sequence()[resi - 1]
            chain, pdb_resnum = pose_res_label(pose, resi)

            # Score global WT once per residue position
            wt_bind  = score_binding(global_wt, scorefxn, resi,
                                     float(args.pack_radius), args.partners, mode)
            wt_total = score_total(global_wt, scorefxn)
            wt_terms = get_per_term_energies(global_wt, scorefxn)

            for t in range(1, int(args.trials) + 1):
                try:
                    for aa in targets:
                        try:
                            # Clone global WT -> mutate -> repack/relax -> score
                            mut_proc = make_mutant(
                                global_wt, resi, aa,
                                float(args.pack_radius), scorefxn,
                                use_fastrelax, relax_scope, float(args.relax_radius),
                                int(args.relax_cycles), mutate_chains,
                            )

                            mut_bind  = score_binding(mut_proc, scorefxn, resi,
                                                      float(args.pack_radius), args.partners, mode)
                            mut_total = score_total(mut_proc, scorefxn)
                            mut_terms = get_per_term_energies(mut_proc, scorefxn)

                            # Binding ΔΔG
                            ddg_bind = float(mut_bind - wt_bind)
                            bind_key = (resi, aa, "bind")
                            ddg_trials[bind_key].append(ddg_bind)
                            avg_bind = float(np.mean(ddg_trials[bind_key]))
                            sd_bind  = float(np.std(ddg_trials[bind_key])) if len(ddg_trials[bind_key]) > 1 else 0.0

                            # Folding ΔΔG
                            ddg_fold = float(mut_total - wt_total)
                            fold_key = (resi, aa, "fold")
                            ddg_trials[fold_key].append(ddg_fold)
                            avg_fold = float(np.mean(ddg_trials[fold_key]))
                            sd_fold  = float(np.std(ddg_trials[fold_key])) if len(ddg_trials[fold_key]) > 1 else 0.0

                            # Per-term deltas
                            wt_term_vals  = [wt_terms.get(term_name, 0.0) for term_name in active_terms]
                            mut_term_vals = [mut_terms.get(term_name, 0.0) for term_name in active_terms]
                            delta_term_vals = [
                                round(mut_terms.get(term_name, 0.0) - wt_terms.get(term_name, 0.0), 4)
                                for term_name in active_terms
                            ]

                            w.writerow([
                                resi, chain, pdb_resnum,
                                wt_aa, aa, t,
                                f"{wt_bind:.3f}", f"{mut_bind:.3f}", f"{ddg_bind:.3f}",
                                f"{avg_bind:.3f}", f"{sd_bind:.3f}",
                                f"{wt_total:.3f}", f"{mut_total:.3f}", f"{ddg_fold:.3f}",
                                f"{avg_fold:.3f}", f"{sd_fold:.3f}",
                                mode, relax_scope, "".join(sorted(mutate_chains)),
                            ] + wt_term_vals + mut_term_vals + delta_term_vals)

                            safe_flush(fh)

                            sys.stderr.write(
                                f"[ok] {mode} {chain}{pdb_resnum} {wt_aa}->{aa} trial {t} "
                                f"ddg_bind={ddg_bind:+.3f} ddg_fold={ddg_fold:+.3f}\n"
                            )

                            # PDB output
                            if should_save(chain, pdb_resnum, aa):
                                pdb_dir.mkdir(parents=True, exist_ok=True)
                                tag = f"{chain}{pdb_resnum}_{wt_aa}to{aa}_trial{t}"
                                mut_proc.dump_pdb(str(pdb_dir / f"mut_{tag}.pdb"))
                                if save_wt_pdb:
                                    global_wt.dump_pdb(str(pdb_dir / f"wt_{chain}{pdb_resnum}.pdb"))

                        except Exception as e:
                            sys.stderr.write(
                                f"[warn] Failed at pose {resi} {wt_aa}->{aa} trial {t}: {e}\n"
                            )
                            safe_flush(fh)

                except Exception as e:
                    sys.stderr.write(
                        f"[warn] Unexpected error at pose {resi} trial {t}: {e}\n"
                    )
                    safe_flush(fh)

    print(f"Streaming results written to: {out_stream}")
    if pdb_dir and pdb_dir.exists():
        print(f"PDB files written to: {pdb_dir}/")


if __name__ == "__main__":
    main()
