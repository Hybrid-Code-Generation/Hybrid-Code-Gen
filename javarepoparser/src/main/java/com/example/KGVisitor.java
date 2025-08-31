package com.example;

import java.util.logging.Logger;

import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.TypeParameter;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class KGVisitor extends VoidVisitorAdapter<Void> {
    private final Neo4jWriter writer;
    private static final Logger logger = Logger.getLogger(KGVisitor.class.getName());
    private static final String logPrefix = "KGVisitor: ";

    public KGVisitor(Neo4jWriter writer) {
        this.writer = writer;
    }

    @Override
    public void visit(MethodCallExpr n, Void arg) {
        logger.info(logPrefix + "Visiting method call: " + n.getNameAsString());
        writer.createCallRelation(n); // This will create the CALLS relationship
        logger.info(logPrefix + "Method call processed: " + n.getNameAsString());
        super.visit(n, arg);
    }

    @Override
    public void visit(ClassOrInterfaceDeclaration n, Void arg) {
        logger.info(logPrefix + "Visiting class: " + n.getNameAsString());
        writer.createClassNode(n);

        // INHERITS
        for (ClassOrInterfaceType extended : n.getExtendedTypes()) {
            writer.linkClassToClass(n, extended);
        }
        // IMPLEMENTS
        for (ClassOrInterfaceType implemented : n.getImplementedTypes()) {
            writer.linkClassToInterface(n, implemented);
        }
        // HAS_ANNOTATION
        for (AnnotationExpr ann : n.getAnnotations()) {
            writer.createAnnotationNode(ann);
            writer.linkClassToAnnotation(n, ann);
        }
        super.visit(n, arg);
    }

    @Override
    public void visit(MethodDeclaration n, Void arg) {
        logger.info(logPrefix + "Visiting method: " + n.getNameAsString());
        writer.createMethodNode(n);
        // Link method to its class
        logger.info(logPrefix + "Linking method '" + n.getNameAsString() + "' to its class");
        writer.linkMethodToClass(n);

        // RETURNS
        writer.linkMethodToType(n, n.getType());

        // TAKES_PARAM, HAS_PARAMETER
        for (Parameter p : n.getParameters()) {
            writer.createParameterNode(p);
            writer.linkMethodToType(n, p.getType());
            // HAS_PARAMETER
            // (already handled by createParameterNode if you want, or add explicit
            // relation)
        }

        // HAS_ANNOTATION
        for (AnnotationExpr ann : n.getAnnotations()) {
            writer.createAnnotationNode(ann);
            writer.linkMethodToAnnotation(n, ann);
        }

        super.visit(n, arg);
    }

    @Override
    public void visit(ConstructorDeclaration n, Void arg) {
        logger.info(logPrefix + "Visiting constructor: " + n.getNameAsString());
        writer.createConstructorNode(n);
        writer.linkConstructorToClass(n);
        for (Parameter p : n.getParameters()) {
            writer.createParameterNode(p);
            writer.linkParameterToConstructor(n, p);
        }
        for (AnnotationExpr ann : n.getAnnotations()) {
            writer.createAnnotationNode(ann);
        }
        super.visit(n, arg);
    }

    @Override
    public void visit(VariableDeclarator n, Void arg) {
        logger.info(logPrefix + "Visiting variable: " + n.getNameAsString());
        writer.createVariableNode(n);

        // DECLARES_VARIABLE: find enclosing method
        n.findAncestor(MethodDeclaration.class).ifPresent(method -> {
            writer.linkMethodToVariable(method, n);
        });

        super.visit(n, arg);
    }

    @Override
    public void visit(FieldDeclaration n, Void arg) {
        logger.info(logPrefix + "Visiting field declaration: " + n.getVariables().get(0).getNameAsString());
        writer.createFieldNode(n);
        // Field linking is now handled inside createFieldNode
        super.visit(n, arg);
    }

    @Override
    public void visit(EnumDeclaration n, Void arg) {
        logger.info(logPrefix + "Visiting enum: " + n.getNameAsString());
        writer.createEnumNode(n);
        for (AnnotationExpr ann : n.getAnnotations()) {
            writer.createAnnotationNode(ann);
            // You may want to add a linkEnumToAnnotation method
        }
        super.visit(n, arg);
    }

    @Override
    public void visit(Parameter n, Void arg) {
        logger.info(logPrefix + "Visiting parameter: " + n.getNameAsString());
        writer.createParameterNode(n);
        super.visit(n, arg);
    }

    @Override
    public void visit(PackageDeclaration n, Void arg) {
        logger.info(logPrefix + "Visiting package: " + n.getNameAsString());
        writer.createPackageNode(n);
        super.visit(n, arg);
    }

    @Override
    public void visit(ImportDeclaration n, Void arg) {
        logger.info(logPrefix + "Visiting import: " + n.getNameAsString());
        writer.createImportNode(n);
        super.visit(n, arg);
    }

    @Override
    public void visit(TypeParameter n, Void arg) {
        logger.info(logPrefix + "Visiting type parameter: " + n.getNameAsString());
        writer.createTypeNode(n);
        super.visit(n, arg);
    }
}
