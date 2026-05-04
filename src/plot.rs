use crate::benchmark::BenchResult;
use plotters::prelude::*;

/// Plot benchmark results as a bar chart PNG.
pub fn plot_results(results: &[BenchResult], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (1024, 640)).into_drawing_area();
    root.fill(&WHITE)?;

    let n = results.len();
    let max_val = results.iter().map(|r| r.throughput).fold(0.0f64, f64::max) * 1.25;
    let labels: Vec<&str> = results.iter().map(|r| r.label.as_str()).collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Mini-DLLM Benchmark Results",
            ("sans-serif", 24).into_font(),
        )
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(0usize..n, 0f64..max_val)?;

    chart
        .configure_mesh()
        .x_label_formatter(&|&x: &usize| -> String { labels.get(x).unwrap_or(&"?").to_string() })
        .y_desc("Throughput (ops/s)")
        .x_label_style(("sans-serif", 14).into_font())
        .y_label_style(("sans-serif", 12).into_font())
        .draw()?;

    let colors: [RGBColor; 4] = [
        RGBColor(70, 130, 180), // Steel blue
        RGBColor(255, 99, 71),  // Tomato
        RGBColor(50, 205, 50),  // Lime green
        RGBColor(255, 165, 0),  // Orange
    ];

    // Draw bars
    for (i, result) in results.iter().enumerate() {
        let color = colors[i % colors.len()];
        chart.draw_series(std::iter::once(Rectangle::new(
            [(i, 0.0), (i + 1, result.throughput)],
            color.filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}
