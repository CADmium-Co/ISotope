use std::error::Error;

use nalgebra::DMatrix;

use crate::sketch::Sketch;

use super::Solver;

pub struct LevenbergMarquardtSolver {
    max_iterations: usize,
    min_loss: f64,
    step_size: f64,
    pseudo_inverse_eps: f64,
    beta: f64,
}

impl Default for LevenbergMarquardtSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl LevenbergMarquardtSolver {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            min_loss: 1e-10,
            step_size: 1e-1,
            pseudo_inverse_eps: 1e-6,
            beta: 1e-5,
        }
    }

    pub fn new_with_params(
        max_iterations: usize,
        min_loss: f64,
        step_size: f64,
        beta: f64,
    ) -> Self {
        Self {
            max_iterations,
            min_loss,
            step_size,
            pseudo_inverse_eps: 1e-6,
            beta,
        }
    }
}

impl Solver for LevenbergMarquardtSolver {
    fn solve(&self, sketch: &mut Sketch) -> Result<(), Box<dyn Error>> {
        let mut iterations = 0;
        let mut loss_sum = f64::INFINITY;

        while iterations < self.max_iterations && loss_sum > self.min_loss {
            let mut data = sketch.get_data();
            let losses = sketch.get_loss_per_constraint();
            loss_sum = losses.sum();
            let jacobian = sketch.get_jacobian();

            data -= (jacobian.transpose() * jacobian.clone()
                + self.beta * DMatrix::identity(jacobian.ncols(), jacobian.ncols()))
            .clone()
            .pseudo_inverse(self.pseudo_inverse_eps)?
                * &jacobian.transpose()
                * &losses
                * self.step_size;

            sketch.set_data(data);

            iterations += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use crate::{
        examples::test_rectangle_rotated::RotatedRectangleDemo,
        solvers::{levenberg_marquardt::LevenbergMarquardtSolver, Solver},
    };

    #[test]
    pub fn test_levenberg_marquardt_solver() -> Result<(), Box<dyn Error>> {
        let mut rectangle = RotatedRectangleDemo::new()?;

        // Now solve the sketch
        let solver = LevenbergMarquardtSolver::new_with_params(1000, 1e-10, 1e-1, 1e-5);
        solver.solve(&mut rectangle.sketch).unwrap();

        println!("loss: {:?}", rectangle.sketch.get_loss());
        println!("point_a: {:?}", rectangle.point_a.as_ref());
        println!("point_b: {:?}", rectangle.point_b.as_ref());
        println!("point_c: {:?}", rectangle.point_c.as_ref());
        println!("point_d: {:?}", rectangle.point_d.as_ref());
        println!("point_reference: {:?}", rectangle.point_reference.as_ref());

        rectangle.check(1e-1)
    }
}
