package com.clueclust;

import com.clueclust.runner.*;
import com.clueclust.util.Parameters;

/**
 * Main class for the LSHDBSCAN application
 */
public class Main {
    public static void main(String[] args) {
        try {
            // Parse command line arguments
            Parameters params = Parameters.parseArguments(args);

            // Run the appropriate algorithm based on the command
            switch (params.getCommand()) {
                case "lshdbscan":
                    new GeneralRunner().run(params);
                    break;
                case "vanilla":
                    new VanillaDBSCANRunner().run(params);
                    break;
                case "vanillalsh":
                    new VanillaDBSCANLSHRunner().run(params);
                    break;
                case "kmeans":
                    new KMeansRunner().run(params);
                    break;
                case "optimize":
                    new OptimalParameterRunner().run(params);
                    break;
                default:
                    System.out.println("Unknown command: " + params.getCommand());
                    System.out
                            .println("Available commands: lshdbscan, vanilla, vanillalsh, parameter, kmeans, optimize");
                    System.exit(1);
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}