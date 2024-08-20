use std::{cell::RefCell, rc::Rc};

use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tsify")]
use tsify::Tsify;

use crate::{
    constraints::ConstraintLike,
    primitives::{point2::Point2, PrimitiveCell},
};

// This is a sketch constraint that makes the end point of an arc coincident with a point.
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "tsify", derive(Tsify))]
#[cfg_attr(feature = "tsify", tsify(into_wasm_abi, from_wasm_abi))]
pub struct FixPoint {
    point: Rc<RefCell<Point2>>,

    desired_pos: Vector2<f64>,
}

impl FixPoint {
    pub fn new(point: Rc<RefCell<Point2>>, desired_pos: Vector2<f64>) -> Self {
        assert!(desired_pos.iter().all(|x| x.is_finite()));
        Self { point, desired_pos }
    }

    pub fn point(&self) -> Rc<RefCell<Point2>> {
        self.point.clone()
    }

    pub fn set_point(&mut self, point: Rc<RefCell<Point2>>) {
        self.point = point;
    }

    pub fn desired_pos(&self) -> Vector2<f64> {
        self.desired_pos
    }

    pub fn set_desired_pos(&mut self, desired_pos: Vector2<f64>) {
        self.desired_pos = desired_pos;
    }
}

impl ConstraintLike for FixPoint {
    fn references(&self) -> Vec<PrimitiveCell> {
        vec![PrimitiveCell::Point2(self.point.clone())]
    }

    fn loss_value(&self) -> f64 {
        let point = self.point.borrow().data();
        let d = point - self.desired_pos;
        0.5 * d.norm_squared()
    }

    fn update_gradient(&mut self) {
        let point = self.point.borrow().data();
        let d = point - self.desired_pos;

        let grad = d.transpose();
        self.point.borrow_mut().add_to_gradient(grad.as_view());
    }

    fn get_type(&self) -> super::Constraint {
        super::Constraint::FixPoint(self.clone())
    }
}

// Run some tests
#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use nalgebra::Vector2;

    use crate::{
        constraints::{fix_point::FixPoint, ConstraintCell, ConstraintLike},
        sketch::Sketch,
        solvers::{gradient_based_solver::GradientBasedSolver, Solver},
    };

    #[test]
    fn test_fix_point() {
        let mut sketch = Sketch::new();

        let point = sketch.add_point2(1.0, 0.0).unwrap();

        let constr1 = Rc::new(RefCell::new(FixPoint::new(
            point.clone(),
            Vector2::new(1.0, 1.0),
        )));
        sketch
            .add_constraint(ConstraintCell::FixPoint(constr1.clone()))
            .unwrap();

        sketch.check_gradients(1e-6, constr1.clone(), 1e-6);
        let solver = GradientBasedSolver::new();
        solver.solve(&mut sketch).unwrap();

        println!("point: {:?}", point.as_ref().borrow());
        assert!(constr1.borrow().loss_value() < 0.001,);
    }
}
