use std::{cell::RefCell, rc::Rc};

use crate::primitives::Parametric;

pub mod angle_between_points;
pub mod coincident;
pub mod distance;
pub mod fix_point;
pub mod lines;

pub trait Constraint {
    fn references(&self) -> Vec<Rc<RefCell<dyn Parametric>>>;
    fn loss_value(&self) -> f64;
    fn update_gradient(&mut self);
}
