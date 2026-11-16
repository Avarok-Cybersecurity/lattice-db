//! Relation registry changes that have to reach the WAL.
//!
//! Edges store a `u16` relation id; the name lives in the collection config.
//! A relation first used by a durable write is logged as
//! [`WalEntry::DefineRelation`](super::WalEntry::DefineRelation) in the same
//! commit as the edge, so replay restores the name along with the edge.

use crate::error::{LatticeError, LatticeResult};
use crate::types::collection::CollectionConfig;

/// The relation id a write of `relation` will use, and whether it is new
///
/// A new relation gets the id the in-memory engines assign
/// (`relations.len()`). That id is refused if another name already holds it,
/// rather than letting two names share one id.
pub fn plan_relation(config: &CollectionConfig, relation: &str) -> LatticeResult<(u16, bool)> {
    if let Some(id) = config.relation_id(relation) {
        return Ok((id, false));
    }
    let id = u16::try_from(config.relations.len()).map_err(|_| LatticeError::InvalidOperation {
        message: format!("relation '{}' exceeds the u16 relation id space", relation),
    })?;
    if let Some(holder) = config.relation_name(id) {
        return Err(LatticeError::InvalidOperation {
            message: format!(
                "relation '{}' would take id {} already held by '{}'",
                relation, id, holder
            ),
        });
    }
    Ok((id, true))
}

/// Register `name` under `relation_id`, refusing a conflicting mapping
pub fn define_relation(
    config: &mut CollectionConfig,
    name: &str,
    relation_id: u16,
) -> LatticeResult<()> {
    match config.relation_id(name) {
        Some(id) if id == relation_id => return Ok(()),
        Some(id) => {
            return Err(relation_conflict(format!(
                "relation '{}' is id {} but the log defines it as {}",
                name, id, relation_id
            )))
        }
        None => {}
    }
    if let Some(holder) = config.relation_name(relation_id) {
        return Err(relation_conflict(format!(
            "relation id {} belongs to '{}' but the log defines it as '{}'",
            relation_id, holder, name
        )));
    }
    config.relations.insert(name.to_string(), relation_id);
    Ok(())
}

fn relation_conflict(context: String) -> LatticeError {
    LatticeError::DataCorruption {
        code: 60010,
        context,
    }
}
