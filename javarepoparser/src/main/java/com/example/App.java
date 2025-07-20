package com.example;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * This is the main application class.
 */
public class App {

    public static void main(String[] args) throws IOException {
        LoggerConfig.setup();

        Logger logger = Logger.getLogger(App.class.getName());
        logger.info("JavaKGBuilder started");
        if (args.length < 1) {
            logger.severe("Usage: java JavaKGBuilder <path-to-java-project>");
            return;
        }
        String projectDir = args[0];
        ProjectParser parser = new ProjectParser(projectDir);

        parser.parse();
    }
}
