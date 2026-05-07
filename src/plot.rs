use crate::benchmark::BenchResult;
use plotters::prelude::*;

/// Plot benchmark results as a horizontal bar chart PNG.
///
/// Each bar is colored per `BenchResult::color` with throughput + μs/step annotation.
pub fn plot_results(results: &[BenchResult], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let n = results.len();
    if n == 0 {
        let root = BitMapBackend::new(path, (800, 200)).into_drawing_area();
        root.fill(&WHITE)?;
        let style = TextStyle::from(("sans-serif", 18).into_font()).color(&BLACK);
        root.draw_text("No benchmark results", &style, (10, 80))?;
        root.present()?;
        return Ok(());
    }

    let max_tp = results.iter().map(|r| r.throughput).fold(0.0f64, f64::max);
    if max_tp <= 0.0 {
        let root = BitMapBackend::new(path, (800, 200)).into_drawing_area();
        root.fill(&WHITE)?;
        root.present()?;
        return Ok(());
    }

    // Dynamic sizing: wider for labels, taller for more results
    let bar_px = 30;
    let gap_px = 10;
    let img_w = 1100;
    let img_h: u32 = (70 + n * (bar_px + gap_px) + 30) as u32;
    let max_val = max_tp * 1.45; // 45% room for annotations

    let root = BitMapBackend::new(path, (img_w, img_h)).into_drawing_area();
    root.fill(&WHITE)?;

    // Y axis: bar i centered at integer y=i, labels at each integer tick
    let y_lo = -0.5f64;
    let y_hi = n as f64 - 0.5;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "MicroGPT-RS Benchmark Results",
            ("sans-serif", 22).into_font(),
        )
        .margin(10)
        .y_label_area_size(200)
        .x_label_area_size(45)
        .build_cartesian_2d(0f64..max_val, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .y_label_formatter(&|&y: &f64| -> String {
            let idx = y.round() as i64;
            if idx >= 0 && (idx as usize) < n {
                results[idx as usize].label.clone()
            } else {
                String::new()
            }
        })
        .y_labels(n)
        .x_desc("Throughput (ops/s)")
        .x_label_style(("sans-serif", 11).into_font())
        .y_label_style(("sans-serif", 11).into_font())
        .draw()?;

    for (i, result) in results.iter().enumerate() {
        let y = i as f64;
        let (cr, cg, cb) = result.color;
        let color = RGBColor(cr, cg, cb);

        // Draw horizontal bar (slightly narrower than full slot for visual gap)
        chart.draw_series(std::iter::once(Rectangle::new(
            [(0.0, y - 0.4), (result.throughput, y + 0.4)],
            color.filled(),
        )))?;

        // Format throughput with unit suffix
        let tp = result.throughput;
        let tp_str = if tp >= 1_000_000.0 {
            let m = tp / 1_000_000.0;
            format!("{m:.2}M")
        } else if tp >= 1_000.0 {
            let k = tp / 1_000.0;
            format!("{k:.0}K")
        } else {
            format!("{tp:.0}")
        };

        // Annotation: throughput + latency
        let us = result.time_per_step_us;
        let label = format!("{tp_str}  ({us:.2} μs)");

        // Place text just past the bar end
        let text_x = result.throughput + max_val * 0.012;
        chart.draw_series(std::iter::once(Text::new(
            label,
            (text_x, y),
            ("sans-serif", 12).into_font(),
        )))?;
    }

    root.present()?;
    Ok(())
}
