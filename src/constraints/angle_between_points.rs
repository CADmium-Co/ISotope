use std::{cell::RefCell, rc::Rc};

use crate::{
    constraints::Constraint,
    primitives::point2::Point2,
};

// This is a sketch constraint that makes the end point of an arc coincident with a point.
#[derive(Debug)]
pub struct AngleBetweenPoints {
    point1: Rc<RefCell<Point2>>,
    point2: Rc<RefCell<Point2>>,
    middle_point: Rc<RefCell<Point2>>,

    desired_angle: f64,
}

impl AngleBetweenPoints {
    pub fn new(
        point1: Rc<RefCell<Point2>>,
        point2: Rc<RefCell<Point2>>,
        middle_point: Rc<RefCell<Point2>>,
        desired_angle: f64,
) -> Self {
        Self {
            point1,
            point2,
            middle_point,
            desired_angle,
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

    pub fn middle_point(&self) -> Rc<RefCell<Point2>> {
        self.middle_point.clone()
    }

    pub fn set_middle_point(&mut self, middle_point: Rc<RefCell<Point2>>) {
        self.middle_point = middle_point;
    }

    pub fn desired_angle(&self) -> f64 {
        self.desired_angle
    }

    pub fn set_desired_angle(&mut self, desired_angle: f64) {
        self.desired_angle = desired_angle;
    }

    pub fn current_angle(&self) -> f64 {
        let point1 = self.point1.borrow().data();
        let point2 = self.point2.borrow().data();
        let middle_point = self.middle_point.borrow().data();

        let d1 = point1 - middle_point;
        let d2 = point2 - middle_point;

        let dot_product = d1.dot(&d2);
        let norm1 = d1.norm();
        let norm2 = d2.norm();

        let cos_theta = dot_product / (norm1 * norm2);
        cos_theta.clamp(-1.0, 1.0).acos()
    }
}

impl Constraint for AngleBetweenPoints {
    fn loss_value(&self) -> f64 {
        let theta = self.current_angle();

        0.5 * (theta - self.desired_angle) * (theta - self.desired_angle)
    }

    fn update_gradient(&mut self) {
        let point1 = self.point1.borrow().data();
        let point2 = self.point2.borrow().data();
        let middle_point = self.middle_point.borrow().data();

        let d1 = point1 - middle_point;
        let d2 = point2 - middle_point;

        let dot_product = d1.dot(&d2);
        let norm1 = d1.norm();
        let norm2 = d2.norm();

        let cos_theta = dot_product / (norm1 * norm2);
        let theta = cos_theta.clamp(-1.0, 1.0).acos();

        let grad_point1 = self.point1.borrow().gradient();
        let grad_point2 = self.point2.borrow().gradient();
        let grad_middle_point = self.middle_point.borrow().gradient();

        let grad_dot_product_from_d1 = d2.transpose();
        let grad_dot_product_from_d2 = d1.transpose();
        let grad_norm_from_d1 = d1.transpose();
        let grad_norm_to_d2 = d2.transpose();

        let grad_cos_theta_from_dot_product = 1.0 / (norm1 * norm2);
        let grad_cos_theta_from_norm1 = -dot_product / (norm1 * norm1 * norm2);
        let grad_cos_theta_from_norm2 = -dot_product / (norm1 * norm2 * norm2);

        let grad_theta_from_cos_theta = -1.0 / (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

        let grad_loss = theta - self.desired_angle;

        let grad_from_d1 = grad_loss * grad_theta_from_cos_theta * grad_cos_theta_from_dot_product * grad_dot_product_from_d1
            + grad_loss * grad_theta_from_cos_theta * grad_cos_theta_from_norm1 * grad_norm_from_d1;
        let grad_from_d2 = grad_loss * grad_theta_from_cos_theta * grad_cos_theta_from_dot_product * grad_dot_product_from_d2
            + grad_loss * grad_theta_from_cos_theta * grad_cos_theta_from_norm2 * grad_norm_to_d2;

        self.point1.borrow_mut().add_to_gradient((grad_from_d1 * grad_point1).as_view());
        self.point2.borrow_mut().add_to_gradient((grad_from_d2 * grad_point2).as_view());
        self.middle_point.borrow_mut().add_to_gradient((-grad_from_d1 * grad_middle_point - grad_from_d2 * grad_middle_point).as_view());
    }
}

// Run some tests
#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        constraints::angle_between_points::AngleBetweenPoints, primitives::point2::Point2, sketch::Sketch,
        constraints::Constraint,
    };

    #[test]
    fn test_vertical_line() {
        let mut sketch = Sketch::new();

        let point_a = Rc::new(RefCell::new(Point2::new(1.0, 0.0)));
        let point_b = Rc::new(RefCell::new(Point2::new(0.0, 1.0)));
        let point_middle = Rc::new(RefCell::new(Point2::new(0.0, 0.0)));
        sketch.add_primitive(point_a.clone());
        sketch.add_primitive(point_b.clone());
        sketch.add_primitive(point_middle.clone());

        let constr1 = Rc::new(RefCell::new(AngleBetweenPoints::new(point_a.clone(), point_b.clone(), point_middle.clone(), std::f64::consts::PI / 4.0)));
        sketch.add_constraint(constr1.clone());

        println!("current angle: {}", constr1.borrow().current_angle() * 180.0 / std::f64::consts::PI);
        sketch.solve(0.001, 100000);

        println!("point_a: {:?}", point_a.as_ref().borrow());
        println!("point_b: {:?}", point_b.as_ref().borrow());
        println!("point_middle: {:?}", point_middle.as_ref().borrow());

        println!("current angle: {}", constr1.borrow().current_angle() * 180.0 / std::f64::consts::PI);

        assert!(
            constr1.borrow().loss_value() < 0.001,
        );
        assert!(false);
    }
}