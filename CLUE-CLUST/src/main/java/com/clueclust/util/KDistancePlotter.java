package com.clueclust.util;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;

/**
 * Utility class for visualizing k-distance plots to help with parameter
 * selection
 */
public class KDistancePlotter {
    // Constants for plot appearance
    private static final int PLOT_WIDTH = 800;
    private static final int PLOT_HEIGHT = 600;
    private static final int MARGIN_LEFT = 80;
    private static final int MARGIN_RIGHT = 50;
    private static final int MARGIN_TOP = 50;
    private static final int MARGIN_BOTTOM = 80;
    private static final int PLOT_AREA_WIDTH = PLOT_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
    private static final int PLOT_AREA_HEIGHT = PLOT_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

    // Colors for different k values
    private static final Color[] SERIES_COLORS = {
            new Color(31, 119, 180), // Blue
            new Color(255, 127, 14), // Orange
            new Color(44, 160, 44), // Green
            new Color(214, 39, 40), // Red
            new Color(148, 103, 189) // Purple
    };

    /**
     * Create a k-distance plot from CSV data
     * 
     * @param csvFile    Input CSV file with k-distance data
     * @param outputFile Output PNG image file
     * @param title      Title of the plot
     * @throws IOException If an I/O error occurs
     */
    public static void createKDistancePlot(String csvFile, String outputFile, String title) throws IOException {
        // Read data from CSV
        Map<Integer, List<Double>> kDistances = readKDistanceData(csvFile);
        if (kDistances.isEmpty()) {
            throw new IOException("No k-distance data found in file: " + csvFile);
        }

        // Create the plot image
        BufferedImage image = new BufferedImage(PLOT_WIDTH, PLOT_HEIGHT, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = image.createGraphics();

        // Configure rendering for better quality
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Draw white background
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, PLOT_WIDTH, PLOT_HEIGHT);

        // Draw plot title
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("SansSerif", Font.BOLD, 16));
        int titleWidth = g2d.getFontMetrics().stringWidth(title);
        g2d.drawString(title, (PLOT_WIDTH - titleWidth) / 2, 30);

        // Calculate data ranges for scaling
        double maxX = 0;
        double maxY = 0;

        for (Map.Entry<Integer, List<Double>> entry : kDistances.entrySet()) {
            List<Double> distances = entry.getValue();
            maxX = Math.max(maxX, distances.size() - 1);

            for (Double distance : distances) {
                maxY = Math.max(maxY, distance);
            }
        }

        // Add 10% margin to max Y for better visualization
        maxY *= 1.1;

        // Draw axes
        g2d.setColor(Color.BLACK);
        g2d.setStroke(new BasicStroke(2));

        // X-axis
        g2d.drawLine(MARGIN_LEFT, PLOT_HEIGHT - MARGIN_BOTTOM,
                PLOT_WIDTH - MARGIN_RIGHT, PLOT_HEIGHT - MARGIN_BOTTOM);

        // Y-axis
        g2d.drawLine(MARGIN_LEFT, MARGIN_TOP,
                MARGIN_LEFT, PLOT_HEIGHT - MARGIN_BOTTOM);

        // Draw axis labels
        g2d.setFont(new Font("SansSerif", Font.BOLD, 12));

        // X-axis label
        String xLabel = "Point index (sorted by distance)";
        int xLabelWidth = g2d.getFontMetrics().stringWidth(xLabel);
        g2d.drawString(xLabel,
                MARGIN_LEFT + (PLOT_AREA_WIDTH - xLabelWidth) / 2,
                PLOT_HEIGHT - 20);

        // Y-axis label
        String yLabel = "Distance";
        g2d.rotate(-Math.PI / 2);
        g2d.drawString(yLabel,
                -MARGIN_TOP - (PLOT_AREA_HEIGHT + g2d.getFontMetrics().stringWidth(yLabel)) / 2,
                25);
        g2d.rotate(Math.PI / 2);

        // Draw tick marks and grid lines for X axis
        g2d.setFont(new Font("SansSerif", Font.PLAIN, 10));
        g2d.setStroke(new BasicStroke(1, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER,
                10.0f, new float[] { 5.0f }, 0.0f));

        int numXTicks = 10;
        for (int i = 0; i <= numXTicks; i++) {
            int x = MARGIN_LEFT + (i * PLOT_AREA_WIDTH) / numXTicks;
            int tickValue = (int) ((i * maxX) / numXTicks);

            // Tick mark
            g2d.setColor(Color.BLACK);
            g2d.drawLine(x, PLOT_HEIGHT - MARGIN_BOTTOM, x, PLOT_HEIGHT - MARGIN_BOTTOM + 5);

            // Tick label
            String tickLabel = String.valueOf(tickValue);
            int labelWidth = g2d.getFontMetrics().stringWidth(tickLabel);
            g2d.drawString(tickLabel, x - labelWidth / 2, PLOT_HEIGHT - MARGIN_BOTTOM + 20);

            // Grid line
            g2d.setColor(new Color(220, 220, 220));
            g2d.drawLine(x, MARGIN_TOP, x, PLOT_HEIGHT - MARGIN_BOTTOM);
        }

        // Draw tick marks and grid lines for Y axis
        int numYTicks = 10;
        for (int i = 0; i <= numYTicks; i++) {
            int y = PLOT_HEIGHT - MARGIN_BOTTOM - (i * PLOT_AREA_HEIGHT) / numYTicks;
            double tickValue = (i * maxY) / numYTicks;

            // Tick mark
            g2d.setColor(Color.BLACK);
            g2d.drawLine(MARGIN_LEFT - 5, y, MARGIN_LEFT, y);

            // Tick label
            String tickLabel = String.format("%.2f", tickValue);
            g2d.drawString(tickLabel, MARGIN_LEFT - 10 - g2d.getFontMetrics().stringWidth(tickLabel), y + 4);

            // Grid line
            g2d.setColor(new Color(220, 220, 220));
            g2d.drawLine(MARGIN_LEFT, y, PLOT_WIDTH - MARGIN_RIGHT, y);
        }

        // Draw data series
        g2d.setStroke(new BasicStroke(2));

        // Draw each k-distance series
        int colorIndex = 0;
        for (Map.Entry<Integer, List<Double>> entry : kDistances.entrySet()) {
            // int k = entry.getKey();
            List<Double> distances = entry.getValue();
            Color seriesColor = SERIES_COLORS[colorIndex % SERIES_COLORS.length];

            g2d.setColor(seriesColor);

            // Draw the series line
            for (int i = 0; i < distances.size() - 1; i++) {
                int x1 = MARGIN_LEFT + (int) (i * PLOT_AREA_WIDTH / maxX);
                int y1 = PLOT_HEIGHT - MARGIN_BOTTOM - (int) (distances.get(i) * PLOT_AREA_HEIGHT / maxY);
                int x2 = MARGIN_LEFT + (int) ((i + 1) * PLOT_AREA_WIDTH / maxX);
                int y2 = PLOT_HEIGHT - MARGIN_BOTTOM - (int) (distances.get(i + 1) * PLOT_AREA_HEIGHT / maxY);

                g2d.drawLine(x1, y1, x2, y2);
            }

            colorIndex++;
        }

        // Draw legend
        int legendX = PLOT_WIDTH - MARGIN_RIGHT + 10;
        int legendY = MARGIN_TOP;
        int legendWidth = 120;
        int legendHeight = kDistances.size() * 20 + 10;

        // Legend box
        g2d.setColor(new Color(255, 255, 255, 200));
        g2d.fillRect(legendX - 10, legendY, legendWidth, legendHeight);
        g2d.setColor(Color.BLACK);
        g2d.setStroke(new BasicStroke(1));
        g2d.drawRect(legendX - 10, legendY, legendWidth, legendHeight);

        // Legend entries
        g2d.setFont(new Font("SansSerif", Font.PLAIN, 12));
        colorIndex = 0;
        for (Integer k : kDistances.keySet()) {
            Color seriesColor = SERIES_COLORS[colorIndex % SERIES_COLORS.length];
            g2d.setColor(seriesColor);

            // Line segment
            g2d.drawLine(legendX, legendY + 15 + colorIndex * 20,
                    legendX + 30, legendY + 15 + colorIndex * 20);

            // Label
            g2d.setColor(Color.BLACK);
            g2d.drawString("k = " + k, legendX + 40, legendY + 20 + colorIndex * 20);

            colorIndex++;
        }

        // Draw potential elbow points
        ElbowDetector elbowDetector = new ElbowDetector();

        colorIndex = 0;
        for (Map.Entry<Integer, List<Double>> entry : kDistances.entrySet()) {
            // int k = entry.getKey();
            List<Double> distances = entry.getValue();

            // Detect elbow points
            int elbowIndex = elbowDetector.findElbowIndex(distances);
            if (elbowIndex > 0) {
                int x = MARGIN_LEFT + (int) (elbowIndex * PLOT_AREA_WIDTH / maxX);
                int y = PLOT_HEIGHT - MARGIN_BOTTOM - (int) (distances.get(elbowIndex) * PLOT_AREA_HEIGHT / maxY);

                // Draw marker at elbow point
                Color seriesColor = SERIES_COLORS[colorIndex % SERIES_COLORS.length];
                g2d.setColor(seriesColor);
                g2d.fillOval(x - 5, y - 5, 10, 10);

                // Draw line to y-axis to show the recommended epsilon value
                g2d.setStroke(new BasicStroke(1, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER,
                        10.0f, new float[] { 5.0f }, 0.0f));
                g2d.drawLine(MARGIN_LEFT, y, x, y);

                // Add label for the recommended epsilon
                String epsLabel = String.format("ε ≈ %.3f", distances.get(elbowIndex));
                g2d.setFont(new Font("SansSerif", Font.BOLD, 12));
                g2d.drawString(epsLabel, MARGIN_LEFT + 5, y - 5);
            }

            colorIndex++;
        }

        // Dispose of graphics context
        g2d.dispose();

        // Save to file
        ImageIO.write(image, "PNG", new File(outputFile));
        System.out.println("K-distance plot saved to: " + outputFile);
    }

    /**
     * Read k-distance data from CSV file
     */
    private static Map<Integer, List<Double>> readKDistanceData(String csvFile) throws IOException {
        Map<Integer, List<Double>> kDistances = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(csvFile))) {
            // Read header line
            String headerLine = reader.readLine();
            if (headerLine == null) {
                return kDistances;
            }

            // Parse header to get k values
            String[] headers = headerLine.split(",");
            for (int i = 1; i < headers.length; i++) {
                String header = headers[i].trim();
                if (header.endsWith("-distance")) {
                    try {
                        int k = Integer.parseInt(header.substring(0, header.indexOf('-')));
                        kDistances.put(k, new ArrayList<>());
                    } catch (NumberFormatException | IndexOutOfBoundsException e) {
                        System.err.println("Warning: Failed to parse k value from header: " + header);
                    }
                }
            }

            // Read data lines
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length < 2) {
                    continue;
                }

                for (int i = 1; i < values.length && i < headers.length; i++) {
                    String header = headers[i].trim();
                    if (header.endsWith("-distance")) {
                        try {
                            int k = Integer.parseInt(header.substring(0, header.indexOf('-')));
                            double distance = Double.parseDouble(values[i].trim());
                            kDistances.get(k).add(distance);
                        } catch (NumberFormatException | IndexOutOfBoundsException e) {
                            // Skip invalid values
                        }
                    }
                }
            }
        }

        return kDistances;
    }

    /**
     * Main method for testing the plotter directly
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: KDistancePlotter <input-csv-file> <output-png-file> [title]");
            System.exit(1);
        }

        String inputFile = args[0];
        String outputFile = args[1];
        String title = args.length > 2 ? args[2] : "K-Distance Plot";

        try {
            createKDistancePlot(inputFile, outputFile, title);
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}