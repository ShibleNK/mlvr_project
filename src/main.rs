use linfa::prelude::*;
use linfa::Dataset;
use linfa_linear::LinearRegression;
use ndarray::RandomExt;
use ndarray::{Array1, Array2, Axis};
use plotters::prelude::*;
use rand::distributions::Uniform;

fn main() {
    // Step 1: Generate synthetic data
    let num_samples = 100;
    let num_features = 1;

    // Create random input data (x) between -1 and 1
    let x: Array2<f64> = Array2::random((num_samples, num_features), Uniform::new(-1.0, 1.0));

    // Define true weight and bias
    let w_true = 2.0;
    let b_true = 0.5;

    // Generate targe  values y = Xw + b + noise
    let noise: Array2<f64> = Array2::random((num_samples, 1), Uniform:::new(-0.1, 0.1));
    let y: Array1<f64> = (&x * w_true).sum_axis(Axis(1)) + b_true + noise.column(0);

    // Step 2: Create a Dataset of linfa
    let dataset = Dataset::new(x.clone(), y.clone());
    
    // Step 3: Fit the linear regression model
    let model = LinearRegression::default()
        .fit(&dataset)
        .expect("Model training failed");

    // step 4: Predict the target values
    let y_pred = model.predict(dataset.records());

    // Step 5: Evaluate the model performance using Root Mean Squared Error (RMSE)
    let rmse = ((&y_pred - dataset.targets())
        .mapv(|a| a.powi(2))
        .mean()
        .unwrap())
        .sqrt();
    println!("Rot Mean Squared Error (RMSE): {:.4}", rmse);

    // Step 6: Visualize the results (scatter plot of data and single regression line)

}

// Funtions to Visualize the scatter plot of the data and a single regression line
fn plot_results(x: &Array2<f64>, y_true: &Array1<f64>, y_pred: &Array1<f64> -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("linearn_regression_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create a chart with labels, grid and caption
    let mut chart = ChartBuilder::on(&root)
        .caption("Linear Regression Results:", ("sans-serif", 30).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.5..1.5, 0.0..3.0)?;

    // Configure the grid and labels
    chart.confgure_mesh()
        .x_desc("X Values")
        .y_desc("Y Values")
        .draw()?;

    // Scatter plot for actual data points (red dots)
    chart.draw_series(
        y_true.iter().zip(x.axis_iter(Axis(0))).map(|(&y_val, x_val)| {
            Circle::new(x_val[0], y_val), 3, ShapeStyle::from(&RED).filled()
        }),
    )?;

    // Get the minimum and maximum values of x
    let x_min = *x.column(0).iter().min_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    let x_max = *x.column(0).iter().max_by(|a,b| a.partial_cmp(b).unwrap();

        // Draw a single regression line (blue line)
        chart.draw_series(LineSeries::new(
                vec![(x_min, y_min), (x_max, y_max)],
                &BLUE,
        ))?;

        // Save the plot to file
        root.present()?;
        println!("Plot saved to 'linearn_regression_plot.png'");

        ok(())
}
