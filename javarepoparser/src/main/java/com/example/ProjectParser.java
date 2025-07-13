package com.example;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.SourceRoot;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.logging.Logger;

public class ProjectParser {
    private final String projectDir;
    private final Neo4jWriter writer;
    private static final Logger logger = Logger.getLogger(ProjectParser.class.getName());

    public ProjectParser(String projectDir) {
        this.projectDir = projectDir;
        this.writer = new Neo4jWriter();
    }

    public void parse() throws IOException {
        logger.info("Cleaning the database");
        writer.clearDB();

        logger.info("Setting up type solver");
        CombinedTypeSolver typeSolver = new CombinedTypeSolver(
                new ReflectionTypeSolver(),
                new JavaParserTypeSolver(new File(projectDir)));

        logger.info("Setting up symbol resolver");
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(typeSolver);
        StaticJavaParser.getParserConfiguration().setSymbolResolver(symbolSolver);

        logger.info("Starting to parse project at " + projectDir);
        SourceRoot root = new SourceRoot(Paths.get(projectDir));
        root.getParserConfiguration().setSymbolResolver(symbolSolver); // Ensure SourceRoot uses the symbol solver

        logger.info("Parsing Java files in the project");

        root.parse("", (_, _, result) -> {
            result.ifSuccessful(cu -> {
                logger.info("Visiting compilation unit: " + cu.getPrimaryTypeName().orElse("Unknown"));
                new KGVisitor(writer).visit(cu, null);
            });
            return SourceRoot.Callback.Result.DONT_SAVE;
        });

        logger.info("Finished parsing project");

        logger.info("Updating node depths in the database");
        writer.updateAllNodeDepths();
    }
}
