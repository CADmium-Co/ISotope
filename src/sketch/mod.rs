use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use nalgebra::{DMatrix, DVector, Vector2};
use serde::{Deserialize, Serialize};

use crate::constraints::angle_between_points::AngleBetweenPoints;
use crate::constraints::distance::euclidian_distance_between_points::EuclidianDistanceBetweenPoints;
use crate::constraints::fix_point::FixPoint;
use crate::constraints::lines::parallel_lines::ParallelLines;
use crate::constraints::lines::perpendicular_lines::PerpendicularLines;
use crate::constraints::ConstraintCell;
use crate::decompose::face::Face;
use crate::decompose::{decompose_sketch, merge_faces};
use crate::error::ISOTopeError;
use crate::primitives::arc::Arc;
use crate::primitives::line::Line;
use crate::primitives::point2::Point2;
use crate::primitives::{point2, PrimitiveCell};

use super::constraints::ConstraintLike;

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Sketch {
    primitives: BTreeMap<u64, PrimitiveCell>,
    primitives_next_id: u64,
    constraints: VecDeque<ConstraintCell>,
}

impl Sketch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_primitive(&mut self, primitive: PrimitiveCell) -> Result<u64, ISOTopeError> {
        // Make sure all referenced primitives are added to the sketch before the primitive
        for reference in primitive.borrow().references().iter() {
            if !self.primitives.iter().any(|(_, p)| reference == p) {
                return Err(ISOTopeError::MissingSketchReferences);
            }
        }
        // Check that the primitive is not already in the sketch
        if self.primitives.iter().any(|(_, p)| p == &primitive) {
            return Err(ISOTopeError::PrimitiveAlreadyInSketch);
        }
        // Add the primitive to the sketch
        self.primitives.insert(self.primitives_next_id, primitive);
        self.primitives_next_id += 1;

        Ok(self.primitives_next_id - 1)
    }

    pub fn add_point2(&mut self, x: f64, y: f64) -> Result<Rc<RefCell<Point2>>, ISOTopeError> {
        let point = Rc::new(RefCell::new(Point2::new(x, y)));
        self.add_primitive(PrimitiveCell::Point2(point.clone()))?;
        Ok(point)
    }

    pub fn add_line(
        &mut self,
        start: Rc<RefCell<Point2>>,
        end: Rc<RefCell<Point2>>,
    ) -> Result<Rc<RefCell<Line>>, ISOTopeError> {
        let line = Rc::new(RefCell::new(Line::new(start, end)));
        self.add_primitive(PrimitiveCell::Line(line.clone()))?;
        Ok(line)
    }

    pub fn add_arc(
        &mut self,
        center: Rc<RefCell<Point2>>,
        radius: f64,
        clockwise: bool,
        start_angle: f64,
        end_angle: f64,
    ) -> Result<Rc<RefCell<Arc>>, ISOTopeError> {
        let arc = Rc::new(RefCell::new(Arc::new(
            center,
            radius,
            clockwise,
            start_angle,
            end_angle,
        )));
        self.add_primitive(PrimitiveCell::Arc(arc.clone()))?;
        Ok(arc)
    }

    pub fn constrain_perpendicular_lines(
        &mut self,
        line1: Rc<RefCell<Line>>,
        line2: Rc<RefCell<Line>>,
    ) -> Result<Rc<RefCell<PerpendicularLines>>, ISOTopeError> {
        let perpendicular_lines = Rc::new(RefCell::new(PerpendicularLines::new(line1, line2)));
        self.add_constraint(ConstraintCell::PerpendicularLines(
            perpendicular_lines.clone(),
        ))?;
        Ok(perpendicular_lines)
    }

    pub fn constrain_parallel_lines(
        &mut self,
        line1: Rc<RefCell<Line>>,
        line2: Rc<RefCell<Line>>,
    ) -> Result<Rc<RefCell<ParallelLines>>, ISOTopeError> {
        let parallel_lines = Rc::new(RefCell::new(ParallelLines::new(line1, line2)));
        self.add_constraint(ConstraintCell::ParallelLines(parallel_lines.clone()))?;
        Ok(parallel_lines)
    }

    pub fn constrain_distance_euclidean(
        &mut self,
        point1: Rc<RefCell<Point2>>,
        point2: Rc<RefCell<Point2>>,
        desired_distance: f64,
    ) -> Result<Rc<RefCell<EuclidianDistanceBetweenPoints>>, ISOTopeError> {
        let distance = Rc::new(RefCell::new(EuclidianDistanceBetweenPoints::new(
            point1,
            point2,
            desired_distance,
        )));
        self.add_constraint(ConstraintCell::EuclideanDistance(distance.clone()))?;
        Ok(distance)
    }

    pub fn constrain_fix_point(
        &mut self,
        point: Rc<RefCell<Point2>>,
        desired_pos: Vector2<f64>,
    ) -> Result<Rc<RefCell<FixPoint>>, ISOTopeError> {
        let fix_point = Rc::new(RefCell::new(FixPoint::new(point, desired_pos)));
        self.add_constraint(ConstraintCell::FixPoint(fix_point.clone()))?;
        Ok(fix_point)
    }

    pub fn constrain_angle_between_points(
        &mut self,
        point1: Rc<RefCell<Point2>>,
        point2: Rc<RefCell<Point2>>,
        middle_point: Rc<RefCell<Point2>>,
        desired_angle: f64,
    ) -> Result<Rc<RefCell<AngleBetweenPoints>>, ISOTopeError> {
        let angle = Rc::new(RefCell::new(AngleBetweenPoints::new(
            point1,
            point2,
            middle_point,
            desired_angle,
        )));
        self.add_constraint(ConstraintCell::AngleBetweenPoints(angle.clone()))?;
        Ok(angle)
    }

    pub fn get_num_primitives(&self) -> usize {
        self.primitives.len()
    }

    pub fn add_constraint(&mut self, constraint: ConstraintCell) -> Result<(), ISOTopeError> {
        // Make sure all referenced primitives are added to the sketch before the constraint
        for reference in constraint.borrow().references().iter() {
            if !self.primitives.iter().any(|(_, p)| p == reference) {
                return Err(ISOTopeError::MissingSketchReferences);
            }
        }
        // Make sure the constraint is not already in the sketch
        if self.constraints.iter().any(|c| c == &constraint) {
            return Err(ISOTopeError::ConstraintAlreadyInSketch);
        }

        self.constraints.push_back(constraint);

        Ok(())
    }

    pub fn get_num_constraints(&self) -> usize {
        self.constraints.len()
    }

    pub fn delete_primitive(&mut self, id: u64) -> Result<(), ISOTopeError> {
        if self.primitives.remove(&id).is_none() {
            return Err(ISOTopeError::PrimitiveNotFound(id));
        }

        Ok(())
    }

    pub fn delete_constraint(&mut self, constraint: ConstraintCell) -> Result<(), ISOTopeError> {
        let init_len = self.constraints.len();
        self.constraints.retain(|c| c == &constraint);

        if init_len == self.constraints.len() {
            return Err(ISOTopeError::ConstraintNotFound);
        }

        Ok(())
    }

    pub fn primitives(&self) -> BTreeMap<u64, PrimitiveCell> {
        self.primitives.clone()
    }

    pub fn constraints(&self) -> VecDeque<ConstraintCell> {
        self.constraints.clone()
    }

    pub fn get_n_dofs(&self) -> usize {
        let mut n_dofs = 0;
        for primitive in self.primitives.iter() {
            n_dofs += primitive.1.borrow().get_data().len();
        }
        n_dofs
    }

    pub fn get_data(&self) -> DVector<f64> {
        let mut data = DVector::zeros(self.get_n_dofs());
        let mut i = 0;
        for primitive in self.primitives.iter() {
            let p = primitive.1.borrow();
            let primitive_data = p.get_data();
            data.rows_mut(i, primitive_data.len())
                .copy_from(&primitive_data);
            i += primitive_data.len();
        }
        data
    }

    pub fn get_loss(&mut self) -> f64 {
        let mut loss = 0.0;
        for constraint in self.constraints.iter_mut() {
            loss += constraint.borrow().loss_value();
        }
        loss
    }

    pub fn get_gradient(&mut self) -> DVector<f64> {
        for primitive in self.primitives.iter_mut() {
            primitive.1.borrow_mut().zero_gradient();
        }

        for constraint in self.constraints.iter_mut() {
            constraint.borrow_mut().update_gradient();
        }

        let mut gradient = DVector::zeros(self.get_n_dofs());
        let mut i = 0;
        for primitive in self.primitives.iter() {
            let p = primitive.1.borrow();
            let primitive_gradient = p.get_gradient();
            assert!(
                primitive_gradient.iter().all(|x| x.is_finite()),
                "Gradient contains NaN or Inf"
            );
            gradient
                .rows_mut(i, primitive_gradient.len())
                .copy_from(&primitive_gradient);
            i += primitive_gradient.len();
        }
        gradient
    }

    pub fn get_loss_per_constraint(&self) -> DVector<f64> {
        let mut loss_per_constraint = DVector::zeros(self.constraints.len());
        for (i, constraint) in self.constraints.iter().enumerate() {
            loss_per_constraint[i] = constraint.borrow().loss_value();
        }
        loss_per_constraint
    }

    pub fn get_jacobian(&self) -> DMatrix<f64> {
        let mut jacobian = DMatrix::zeros(self.constraints.len(), self.get_n_dofs());
        for (i, constraint) in self.constraints.iter().enumerate() {
            // Zero the gradients of all primitives
            for primitive in self.primitives.iter() {
                primitive.1.borrow_mut().zero_gradient();
            }
            // Update the gradient of the constraint
            constraint.borrow_mut().update_gradient();
            // Copy the gradient of the constraint to the jacobian
            let mut j = 0;
            for primitive in self.primitives.iter() {
                let p = primitive.1.borrow();
                let primitive_gradient = p.get_gradient();
                jacobian
                    .row_mut(i)
                    .columns_mut(j, primitive_gradient.len())
                    .copy_from(&primitive_gradient.transpose());
                j += primitive_gradient.len();
            }
        }
        jacobian
    }

    pub fn set_data(&mut self, data: DVector<f64>) {
        assert!(data.len() == self.get_n_dofs());
        let mut i = 0;
        for primitive in self.primitives.iter_mut() {
            let n = primitive.1.borrow().get_data().len();
            primitive.1.borrow_mut().set_data(data.rows(i, n).as_view());
            i += n;
        }
    }

    // This function is used in test cases to check the gradients of the primitives
    pub fn check_gradients(
        &mut self,
        epsilon: f64,
        constraint: Rc<RefCell<impl ConstraintLike>>,
        check_epsilon: f64,
    ) {
        // Update all gradients
        self.get_gradient();

        // Compare to numerical gradients
        let constraint_loss = constraint.borrow().loss_value();
        for primitive in self.primitives.iter_mut() {
            let original_value = primitive.1.borrow().get_data().clone_owned();
            let analytical_gradient = primitive.1.borrow().get_gradient().clone_owned();
            let mut numerical_gradient = DVector::zeros(original_value.len());
            let n = primitive.1.borrow().get_data().len();
            assert!(n == analytical_gradient.len());
            for i in 0..n {
                let mut new_value = original_value.clone_owned();
                new_value[i] += epsilon;
                primitive
                    .1
                    .borrow_mut()
                    .set_data(new_value.clone().as_view());
                let new_loss = constraint.borrow().loss_value();
                primitive
                    .1
                    .borrow_mut()
                    .set_data(original_value.clone().as_view());
                numerical_gradient[i] = (new_loss - constraint_loss) / epsilon;
            }

            println!("Analytical gradient: {:?}", analytical_gradient);
            println!("Numerical gradient: {:?}", numerical_gradient);

            let error = (numerical_gradient - analytical_gradient).norm();
            println!("Error: {}", error);
            assert!(error < check_epsilon);
        }
    }

    // Helper functions
    pub fn get_all_points(&self) -> BTreeMap<u64, point2::Point2> {
        self.primitives
            .iter()
            .filter_map(|(k, p)| {
                if let super::primitives::Primitive::Point2(point) = p.borrow().to_primitive() {
                    Some((*k, point))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_primitive_id(&self, primitive: &PrimitiveCell) -> Option<u64> {
        self.primitives
            .iter()
            .find(|(_, p)| primitive == *p)
            .map(|(k, _)| *k)
    }

    pub fn get_primitive_by_id(&self, id: u64) -> Option<&PrimitiveCell> {
        self.primitives.get(&id)
    }

    pub fn get_faces(&self) -> Vec<Face> {
        decompose_sketch(self)
    }

    pub fn get_merged_faces(&self) -> Vec<Face> {
        merge_faces(self.get_faces())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        constraints::coincident::arc_end_point_coincident::ArcEndPointCoincident,
        examples::test_rectangle_rotated::RotatedRectangleDemo,
        primitives::{arc::Arc, point2::Point2},
    };

    use super::*;

    #[test]
    fn test_references_have_to_be_added_beforehand() {
        assert!(std::panic::catch_unwind(|| {
            let mut sketch = Sketch::new();

            let point = Rc::new(RefCell::new(Point2::new(0.0, 0.0)));
            let arc = Rc::new(RefCell::new(Arc::new(point, 1.0, true, 0.0, 1.0)));

            sketch
                .add_primitive(PrimitiveCell::Arc(arc.clone()))
                .unwrap();
        })
        .is_err());
    }

    #[test]
    fn test_primitive_cannot_be_added_twice() {
        assert!(std::panic::catch_unwind(|| {
            let mut sketch = Sketch::new();

            let point = PrimitiveCell::Point2(Rc::new(RefCell::new(Point2::new(0.0, 0.0))));
            sketch.add_primitive(point.clone()).unwrap();
            sketch.add_primitive(point.clone()).unwrap();
        })
        .is_err());
    }

    #[test]
    fn test_constraint_references_have_to_be_added_beforehand() {
        assert!(std::panic::catch_unwind(|| {
            let mut sketch = Sketch::new();

            let point = Rc::new(RefCell::new(Point2::new(0.0, 0.0)));
            let arc = Rc::new(RefCell::new(Arc::new(point.clone(), 1.0, true, 0.0, 1.0)));

            sketch
                .add_primitive(PrimitiveCell::Point2(point.clone()))
                .unwrap();

            let constraint = Rc::new(RefCell::new(ArcEndPointCoincident::new(arc, point)));
            sketch
                .add_constraint(ConstraintCell::ArcEndPointCoincident(constraint))
                .unwrap();
        })
        .is_err());
    }

    #[test]
    fn test_constraint_cannot_be_added_twice() {
        assert!(std::panic::catch_unwind(|| {
            let mut sketch = Sketch::new();

            let point = sketch.add_point2(0.0, 0.0).unwrap();
            let arc = sketch.add_arc(point.clone(), 1.0, true, 0.0, 1.0).unwrap();

            let constraint = Rc::new(RefCell::new(ArcEndPointCoincident::new(
                arc.clone(),
                point.clone(),
            )));
            sketch
                .add_constraint(ConstraintCell::ArcEndPointCoincident(constraint.clone()))
                .unwrap();
            sketch
                .add_constraint(ConstraintCell::ArcEndPointCoincident(constraint.clone()))
                .unwrap();
        })
        .is_err());
    }

    #[test]
    fn test_data_and_grad_functions() {
        let rect = RotatedRectangleDemo::new().unwrap();
        let mut sketch = rect.sketch;
        sketch.get_data();
        sketch.get_loss();
        sketch.get_gradient();
        sketch.get_loss_per_constraint();
        sketch.get_jacobian();
    }
}
