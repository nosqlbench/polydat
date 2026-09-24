// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Fusion units: how nodes are grouped into native code, one rule for
//! every engine that fuses (SRD-105; engines.md §2, §8).
//!
//! A unit is a connected, convex set of fusible nodes of one class. The
//! interpreter's cone planner fuses such components into cones, the
//! native tier compiles each into one segment, and pure native code
//! compiles each into one block of its function. Grouping by the graph's
//! connections rather than by position is what lets a pull run only its
//! output's cone in native code: two chains that share nothing are two
//! units, so pulling one does not run the other, where a run of
//! consecutive nodes would have fused them (runtime_model.md R2).
//!
//! A component must also be convex: no path may leave it and come back.
//! An eligible→ineligible→eligible sandwich whose ends connect through
//! another eligible path lands both ends in one component while the
//! middle stays out, and fusing it would make the middle both a consumer
//! and a producer of the unit, a cycle between units. A component that
//! is not convex is split where a path that leaves it comes back, and
//! nowhere else (`convex_pieces`).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// The units of a graph, in an order every unit's producers precede.
pub(crate) struct UnitPlan {
    /// Each unit's members, in the preferred order.
    pub(crate) units: Vec<Vec<usize>>,
    /// The unit each node belongs to.
    pub(crate) unit_of: Vec<usize>,
}

/// The connected components of the fusible nodes, joining two nodes
/// across a wire when both are fusible and of one class, each with its
/// members in index order; a node that is not fusible is left out.
pub(crate) fn components(preds: &[Vec<usize>], fusible: &[bool], class: &[u64]) -> Vec<Vec<usize>> {
    lumped_components(preds, fusible, class, &|_| false)
}

/// `components`, with the fusible nodes of every class `lump` names
/// joined whether or not a wire connects them: a class whose units never
/// run by cone (compile constants, folded once at build) gains nothing
/// from being split, and each unit is a function to compile.
fn lumped_components(
    preds: &[Vec<usize>],
    fusible: &[bool],
    class: &[u64],
    lump: &dyn Fn(u64) -> bool,
) -> Vec<Vec<usize>> {
    let n = preds.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..n {
        if !fusible[i] {
            continue;
        }
        for &j in &preds[i] {
            if fusible[j] && class[j] == class[i] {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    let mut first_of: std::collections::HashMap<u64, usize> = Default::default();
    for i in (0..n).filter(|&i| fusible[i] && lump(class[i])) {
        let first = *first_of.entry(class[i]).or_insert(i);
        let (a, b) = (find(&mut parent, i), find(&mut parent, first));
        parent[a] = b;
    }
    let mut by_root: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for i in (0..n).filter(|&i| fusible[i]) {
        by_root.entry(find(&mut parent, i)).or_default().push(i);
    }
    by_root.into_values().collect()
}

/// True when no path leaves `members` and comes back: walk the consumer
/// graph from the members' outside consumers, through nodes that are
/// not members; reaching a member proves a path re-enters.
pub(crate) fn is_convex(members: &[usize], consumers: &[Vec<usize>]) -> bool {
    let n = consumers.len();
    let mut is_member = vec![false; n];
    for &m in members {
        is_member[m] = true;
    }
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    for &m in members {
        for &c in &consumers[m] {
            if !is_member[c] && !seen[c] {
                seen[c] = true;
                stack.push(c);
            }
        }
    }
    while let Some(v) = stack.pop() {
        for &c in &consumers[v] {
            if is_member[c] {
                return false;
            }
            if !seen[c] {
                seen[c] = true;
                stack.push(c);
            }
        }
    }
    true
}

/// Split a component that is not convex into as few convex pieces as its
/// re-entering paths force. A member's stage is how many times a path
/// to it has left the component and come back: the most, over its
/// producers, of the producer's stage, plus one where the producer is
/// outside the component and downstream of it. Stages never fall along
/// a wire, so a path that leaves a stage's members arrives, if it
/// returns, at a later stage; the members of one stage, split by the
/// wires between them, are convex, and no two pieces form a cycle.
/// `topo` is every node in a topological order.
fn convex_pieces(
    members: &[usize],
    preds: &[Vec<usize>],
    topo: &[usize],
    lumped: bool,
) -> Vec<Vec<usize>> {
    let n = preds.len();
    let mut is_member = vec![false; n];
    for &m in members {
        is_member[m] = true;
    }
    // Downstream of the component, or in it.
    let mut reached = vec![false; n];
    let mut stage = vec![0u64; n];
    for &v in topo {
        let mut s = 0;
        let mut r = is_member[v];
        for &p in &preds[v] {
            r |= reached[p];
            let returns = is_member[v] && !is_member[p] && reached[p];
            s = s.max(stage[p] + returns as u64);
        }
        stage[v] = s;
        reached[v] = r;
    }
    lumped_components(preds, &is_member, &stage, &|_| lumped)
}

/// Plan the units of a graph of `preds.len()` nodes.
///
/// `preds[i]` are the nodes `i` reads from. A node that is not
/// `fusible` is a unit of its own, and nodes fuse only within one
/// `class` (a lifecycle, a volatility); the classes `lump` names fuse
/// whole, connected or not. `rank` is each node's position in a
/// preferred topological order: it is the order a component that is not
/// convex is walked in to split it, and it decides which ready unit goes
/// first, so a graph already in a good order keeps it. `inputs[i]` are
/// the kernel inputs node `i` reads, by any consistent id.
pub(crate) fn plan_units(
    preds: &[Vec<usize>],
    inputs: &[Vec<usize>],
    fusible: &[bool],
    class: &[u64],
    rank: &[usize],
    lump: &dyn Fn(u64) -> bool,
) -> UnitPlan {
    let n = preds.len();
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, ps) in preds.iter().enumerate() {
        for &p in ps {
            consumers[p].push(i);
        }
    }
    let mut unit_of = vec![usize::MAX; n];
    let mut units: Vec<Vec<usize>> = Vec::new();
    let mut by_rank = |mut members: Vec<usize>, units: &mut Vec<Vec<usize>>| {
        members.sort_by_key(|&m| rank[m]);
        for &m in &members {
            unit_of[m] = units.len();
        }
        units.push(members);
    };
    // The nodes in the preferred order, to walk the graph topologically.
    let mut at_rank = vec![0usize; n];
    for (i, &r) in rank.iter().enumerate() {
        at_rank[r] = i;
    }
    for component in lumped_components(preds, fusible, class, lump) {
        if component.len() == 1 || is_convex(&component, &consumers) {
            by_rank(component, &mut units);
            continue;
        }
        let lumped = lump(class[component[0]]);
        for piece in convex_pieces(&component, preds, &at_rank, lumped) {
            by_rank(piece, &mut units);
        }
    }
    for i in (0..n).filter(|&i| !fusible[i]) {
        by_rank(vec![i], &mut units);
    }

    // A leaf that reads only kernel inputs and that nothing reads, such
    // as the copy that exposes an input as an output, is connected to
    // nothing, and alone it would be a unit, and a native call, of its
    // own for one copy. It joins the first unit of its class that reads
    // one of the same inputs: no path can leave that unit through it
    // and return, so the unit stays convex, and a pull of the unit's
    // outputs pays one copy more.
    let mut moved = false;
    for m in 0..n {
        if !fusible[m]
            || !preds[m].is_empty()
            || !consumers[m].is_empty()
            || inputs[m].is_empty()
            || units[unit_of[m]].len() != 1
        {
            continue;
        }
        let home = unit_of[m];
        let target = units
            .iter()
            .enumerate()
            .filter(|&(t, members)| {
                t != home
                    && !members.is_empty()
                    && fusible[members[0]]
                    && class[members[0]] == class[m]
                    && members
                        .iter()
                        .any(|&x| inputs[x].iter().any(|i| inputs[m].contains(i)))
            })
            .map(|(t, members)| (rank[members[0]], t))
            .min();
        if let Some((_, t)) = target {
            units[home].clear();
            units[t].push(m);
            units[t].sort_by_key(|&x| rank[x]);
            unit_of[m] = t;
            moved = true;
        }
    }
    if moved {
        units.retain(|members| !members.is_empty());
        for (u, members) in units.iter().enumerate() {
            for &m in members {
                unit_of[m] = u;
            }
        }
    }

    // Order the units: a unit is ready once every unit it reads from
    // has gone, and the ready unit that comes first in the preferred
    // order goes next.
    let u = units.len();
    let first: Vec<usize> = units.iter().map(|m| rank[m[0]]).collect();
    let mut after: Vec<Vec<usize>> = vec![Vec::new(); u];
    let mut waiting = vec![0usize; u];
    for (to, members) in units.iter().enumerate() {
        let mut from: Vec<usize> = members
            .iter()
            .flat_map(|&m| preds[m].iter().map(|&p| unit_of[p]))
            .filter(|&f| f != to)
            .collect();
        from.sort_unstable();
        from.dedup();
        waiting[to] = from.len();
        for f in from {
            after[f].push(to);
        }
    }
    let mut ready: BinaryHeap<Reverse<(usize, usize)>> = (0..u)
        .filter(|&x| waiting[x] == 0)
        .map(|x| Reverse((first[x], x)))
        .collect();
    let mut order: Vec<usize> = Vec::with_capacity(u);
    while let Some(Reverse((_, x))) = ready.pop() {
        order.push(x);
        for &y in &after[x] {
            waiting[y] -= 1;
            if waiting[y] == 0 {
                ready.push(Reverse((first[y], y)));
            }
        }
    }
    assert_eq!(order.len(), u, "units of a convex partition are acyclic");
    let mut renumber = vec![0usize; u];
    for (new, &old) in order.iter().enumerate() {
        renumber[old] = new;
    }
    let mut ordered: Vec<Vec<usize>> = vec![Vec::new(); u];
    for (old, members) in units.into_iter().enumerate() {
        ordered[renumber[old]] = members;
    }
    for x in unit_of.iter_mut() {
        *x = renumber[*x];
    }
    UnitPlan {
        units: ordered,
        unit_of,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(preds: Vec<Vec<usize>>, fusible: Vec<bool>) -> UnitPlan {
        let n = preds.len();
        plan_units(
            &preds,
            &vec![Vec::new(); n],
            &fusible,
            &vec![0; n],
            &(0..n).collect::<Vec<_>>(),
            &|_| false,
        )
    }

    /// Two chains that share nothing are two units, however their nodes
    /// interleave; a run of consecutive nodes would have been one.
    #[test]
    fn independent_chains_are_separate_units() {
        // a0 b0 a1 b1: a1 reads a0, b1 reads b0.
        let p = plan(vec![vec![], vec![], vec![0], vec![1]], vec![true; 4]);
        assert_eq!(p.units.len(), 2);
        assert_eq!(p.unit_of[0], p.unit_of[2]);
        assert_eq!(p.unit_of[1], p.unit_of[3]);
        assert_ne!(p.unit_of[0], p.unit_of[1]);
    }

    /// A fusible node reached around an unfusible one: 0 → 1(unfusible)
    /// → 2, and 0 → 2. Fusing {0, 2} would put 1 both after and before
    /// the unit, so the component splits into runs, and every unit's
    /// producers come before it.
    #[test]
    fn a_component_that_is_not_convex_splits() {
        let p = plan(vec![vec![], vec![0], vec![0, 1]], vec![true, false, true]);
        assert_eq!(p.units.len(), 3);
        for (u, members) in p.units.iter().enumerate() {
            for &m in members {
                for &pr in &[vec![], vec![0], vec![0, 1]][m] {
                    assert!(p.unit_of[pr] <= u, "unit {u} reads a later unit");
                }
            }
        }
    }

    /// A component that is not convex is cut only where a path comes
    /// back: 0 → 1(unfusible) → 5 and 0 → 4 → 5, with an unrelated
    /// chain 2 → 3 written between them. 0 and 4 fuse across the chain,
    /// where a run of consecutive members would have been cut by it, and
    /// 5 is a unit of its own because a path from 0 returns to it
    /// through 1.
    #[test]
    fn a_component_is_cut_only_where_a_path_returns() {
        let preds = vec![
            vec![],     // 0
            vec![0],    // 1, unfusible
            vec![],     // 2: chain head
            vec![2],    // 3: chain
            vec![0],    // 4
            vec![1, 4], // 5
        ];
        let fusible = vec![true, false, true, true, true, true];
        let p = plan_units(
            &preds,
            &vec![Vec::new(); 6],
            &fusible,
            &[0; 6],
            &[0, 1, 2, 3, 4, 5],
            &|_| false,
        );
        assert_eq!(p.unit_of[0], p.unit_of[4], "0 and 4 fuse across the chain");
        assert_ne!(p.unit_of[0], p.unit_of[5], "5 is where the path returns");
        assert_eq!(p.unit_of[2], p.unit_of[3]);
        assert_ne!(p.unit_of[2], p.unit_of[0]);
        assert_eq!(p.units.len(), 4);
    }

    /// The engine ladder's shape: a connected group reading inputs 0 and
    /// 1, and three leaves that each copy one input out and feed
    /// nothing. The leaves over inputs the group reads join it, so a full
    /// evaluation is one unit; one over an input nothing else reads stays
    /// its own.
    #[test]
    fn a_leaf_copying_an_input_joins_a_unit_over_that_input() {
        // 0: reads inputs 0, 1; 1: reads node 0. 2, 3, 4: leaves copying
        // inputs 0, 1, and 9.
        let preds = vec![vec![], vec![0], vec![], vec![], vec![]];
        let inputs = vec![vec![0, 1], vec![], vec![0], vec![1], vec![9]];
        let p = plan_units(
            &preds,
            &inputs,
            &[true; 5],
            &[0; 5],
            &[0, 1, 2, 3, 4],
            &|_| false,
        );
        assert_eq!(p.units.len(), 2, "{:?}", p.units);
        assert_eq!(p.unit_of[2], p.unit_of[0]);
        assert_eq!(p.unit_of[3], p.unit_of[0]);
        assert_ne!(p.unit_of[4], p.unit_of[0]);
    }

    /// Nodes of different classes never share a unit.
    #[test]
    fn classes_do_not_fuse() {
        let preds = vec![vec![], vec![0]];
        let p = plan_units(
            &preds,
            &vec![Vec::new(); 2],
            &[true, true],
            &[0, 1],
            &[0, 1],
            &|_| false,
        );
        assert_eq!(p.units.len(), 2);
    }
}
