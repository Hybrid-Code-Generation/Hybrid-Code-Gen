package com.example;

import java.io.IOException;
import java.util.logging.*;

public class LoggerConfig {
    public static void setup() {
        Logger logger = Logger.getLogger(""); // Root logger

        // Remove default console handlers
        for (Handler handler : logger.getHandlers()) {
            logger.removeHandler(handler);
        }

        try {
            FileHandler fileHandler = new FileHandler("java-kg-builder.log", true); // append = true
            fileHandler.setFormatter(new SimpleFormatter());
            logger.addHandler(fileHandler);
            logger.setLevel(Level.INFO); // You can set FINE, WARNING, etc.

        } catch (IOException e) {
            System.err.println("Failed to setup logger: " + e.getMessage());
        }
    }
}