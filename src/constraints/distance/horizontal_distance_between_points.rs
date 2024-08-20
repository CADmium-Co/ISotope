use std::{cell::RefCell, rc::Rc};

use nalgebra::Matrix1x2;
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
pub struct HorizontalDistanceBetweenPoints {
    point1: Rc<RefCell<Point2>>,
    point2: Rc<RefCell<Point2>>,

    desired_distance: f64,
}

impl HorizontalDistanceBetweenPoints {
    pub fn new(
        point1: Rc<RefCell<Point2>>,
        point2: Rc<RefCell<Point2>>,
        desired_distance: f64,
    ) -> Self {
        assert!(desired_distance.is_finite());
        Self {
            point1,
            point2,
            desired_distance,
        }
    }

    pub fn point1(&self) -> Rc<RefCell<Point2>> {
        self.point1.clone()
    }

    pub fn set_point1(&mut self, point1: Rc<RefCell<Point2>>) {
        self.point1 = point1;
    }

    pub fn point2(&self) -> Rc<RefCell<Point2>> {
        self.point2.clone()
    }

    pub fn set_point2(&mut self, point2: Rc<RefCell<Point2>>) {
        self.point2 = point2;
    }

    pub fn desired_distance(&self) -> f64 {
        self.desired_distance
    }

    pub fn set_desired_distance(&mut self, desired_distance: f64) {
        self.desired_distance = desired_distance;
    }

    pub fn current_distance(&self) -> f64 {
        let point1 = self.point1.borrow().data();
        let point2 = self.point2.borrow().data();

        point2.x - point1.x
    }
}

impl ConstraintLike for HorizontalDistanceBetweenPoints {
    fn references(&self) -> Vec<PrimitiveCell> {
        vec![
            PrimitiveCell::Point2(self.point1.clone()),
            PrimitiveCell::Point2(self.point2.clone()),
        ]
    }

    fn loss_value(&self) -> f64 {
        let distance = self.current_distance();
        let err = distance - self.desired_distance;
        0.5 * err * err
    }

    fn update_gradient(&mut self) {
        let point1 = self.point1.borrow().data();
        let point2 = self.point2.borrow().data();

        let d = point2 - point1;

        let distance = d.x;
        let err = distance - self.desired_distance;
        let _loss = 0.5 * err * err;

        let grad_distance_from_d = Matrix1x2::<f64>::new(1.0, 0.0);
        let grad_err_from_distance = 1.0;
        let grad_loss_from_err = err;

        let grad_from_d = grad_loss_from_err * grad_err_from_distance * grad_distance_from_d;

        let grad_point1 = self.point1.borrow().point_gradient();
        let grad_point2 = self.point2.borrow().point_gradient();

        let grad_from_point1 = -grad_from_d * grad_point1;
        let grad_from_point2 = grad_from_d * grad_point2;

        self.point1
            .borrow_mut()
            .add_to_gradient(grad_from_point1.as_view());
        self.point2
            .borrow_mut()
            .add_to_gradient(grad_from_point2.as_view());
    }

    fn get_type(&self) -> crate::constraints::Constraint {
        crate::constraints::Constraint::HorizontalDistance(self.clone())
    }
}

// Run some tests
#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        constraints::{
            distance::horizontal_distance_between_points::HorizontalDistanceBetweenPoints,
            ConstraintCell, ConstraintLike,
        },
        sketch::Sketch,
        solvers::{gradient_based_solver::GradientBasedSolver, Solver},
    };

    #[test]
    fn test_horizontal_distance_between_points() {
        let mut sketch = Sketch::new();

        let point_a = sketch.add_point2(1.0, 0.0).unwrap();
        let point_b = sketch.add_point2(0.0, 1.0).unwrap();

        let constr1 = Rc::new(RefCell::new(HorizontalDistanceBetweenPoints::new(
            point_a.clone(),
            point_b.clone(),
            -3.0,
        )));
        sketch
            .add_constraint(ConstraintCell::HorizontalDistance(constr1.clone()))
            .unwrap();

        sketch.check_gradients(1e-6, constr1.clone(), 1e-6);
        let solver = GradientBasedSolver::new();
        solver.solve(&mut sketch).unwrap();

        println!("point_a: {:?}", point_a.as_ref().borrow());
        println!("point_b: {:?}", point_b.as_ref().borrow());
        println!(
            "distance: {:?}",
            (point_b.as_ref().borrow().x() - point_a.as_ref().borrow().x())
        );

        assert!(constr1.borrow().loss_value() < 0.001,);
    }
}
