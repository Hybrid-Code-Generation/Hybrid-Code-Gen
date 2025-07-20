package com.example;

import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.LiteralExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.TypeParameter;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import org.neo4j.driver.*;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;

public class Neo4jWriter {
    private final Driver driver;
    private static final Logger logger = Logger.getLogger(Neo4jWriter.class.getName());
    private static final String logPrefix = "Neo4jWriter: ";

    public Neo4jWriter() {
        driver = GraphDatabase.driver("bolt://4.187.169.27:7687", AuthTokens.basic("neo4j", "MyStrongPassword123"));
    }

    public void createClassNode(ClassOrInterfaceDeclaration clazz) {
        String name = clazz.getNameAsString();
        logger.info(logPrefix + "Creating class node: " + name);
        try (Session session = driver.session()) {
            session.run("MERGE (c:Class {name: $name})", Map.of("name", name));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createClassNode: " + e.getMessage());
        }
    }

    public void createMethodNode(MethodDeclaration method) {
        String name = method.getNameAsString();
        logger.info(logPrefix + "Creating method node: " + name);
        try (Session session = driver.session()) {
            session.run("MERGE (m:Method {name: $name})", Map.of("name", name));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createMethodNode: " + e.getMessage());
        }
    }

    public void linkConstructorToClass(ConstructorDeclaration ctor) {
        String ctorName = ctor.getNameAsString();
        String className = ctor.getParentNode()
                .filter(n -> n instanceof ClassOrInterfaceDeclaration)
                .map(n -> ((ClassOrInterfaceDeclaration) n).getNameAsString())
                .orElse(null);
        if (className == null)
            return;
        try (Session session = driver.session()) {
            session.run(
                    "MATCH (c:Class {name: $className}), (ctor:Constructor {name: $ctorName}) " +
                            "MERGE (c)-[:HAS_CONSTRUCTOR]->(ctor)",
                    Map.of("className", className, "ctorName", ctorName));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in linkConstructorToClass: " + e.getMessage());
        }
    }

    public void linkParameterToConstructor(ConstructorDeclaration ctor, Parameter param) {
        String ctorName = ctor.getNameAsString();
        String paramName = param.getNameAsString();
        try (Session session = driver.session()) {
            session.run(
                    "MATCH (ctor:Constructor {name: $ctorName}), (p:Parameter {name: $paramName}) " +
                            "MERGE (ctor)-[:HAS_PARAMETER]->(p)",
                    Map.of("ctorName", ctorName, "paramName", paramName));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in linkParameterToConstructor: " + e.getMessage());
        }
    }

    public void linkMethodToType(MethodDeclaration method, com.github.javaparser.ast.type.Type type) {
        String methodName = method.getNameAsString();
        String typeName = type.toString();
        if (methodName == null || typeName == null) {
            logger.warning(logPrefix + "Cannot link method to type: missing name.");
            return;
        }
        try (Session session = driver.session()) {
            session.run(
                    "MERGE (m:Method {name: $methodName}) " +
                            "MERGE (t:Type {name: $typeName}) " +
                            "MERGE (m)-[:RETURNS]->(t)",
                    Map.of("methodName", methodName, "typeName", typeName));
            logger.info(logPrefix + "Linked method '" + methodName + "' to return type '" + typeName + "'");
        } catch (Exception e) {
            logger.severe(logPrefix + "Failed to link method '" + methodName + "' to type '" + typeName + "': "
                    + e.getMessage());
        }
    }

    public void linkClassToClass(ClassOrInterfaceDeclaration clazz, ClassOrInterfaceType superClazzType) {
        String className = clazz.getNameAsString();
        String superName = superClazzType.getNameAsString();
        if (className == null || superName == null) {
            logger.warning(logPrefix + "Cannot link class to superclass: missing name.");
            return;
        }
        try (Session session = driver.session()) {
            session.run(
                    "MERGE (c:Class {name: $className}) " +
                            "MERGE (s:Class {name: $superName}) " +
                            "MERGE (c)-[:INHERITS]->(s)",
                    Map.of("className", className, "superName", superName));
            logger.info(logPrefix + "Linked class '" + className + "' to superclass '" + superName + "'");
        } catch (Exception e) {
            logger.severe(logPrefix + "Failed to link class '" + className + "' to superclass '" + superName + "': "
                    + e.getMessage());
        }
    }

    public void linkClassToInterface(ClassOrInterfaceDeclaration clazz, ClassOrInterfaceType ifaceType) {
        String className = clazz.getNameAsString();
        String ifaceName = ifaceType.getNameAsString();
        if (className == null || ifaceName == null) {
            logger.warning(logPrefix + "Cannot link class to interface: missing name.");
            return;
        }
        try (Session session = driver.session()) {
            session.run(
                    "MERGE (c:Class {name: $className}) " +
                            "MERGE (i:Interface {name: $ifaceName}) " +
                            "MERGE (c)-[:IMPLEMENTS]->(i)",
                    Map.of("className", className, "ifaceName", ifaceName));
            logger.info(logPrefix + "Linked class '" + className + "' to interface '" + ifaceName + "'");
        } catch (Exception e) {
            logger.severe(logPrefix + "Failed to link class '" + className + "' to interface '" + ifaceName + "': "
                    + e.getMessage());
        }
    }

    public void linkMethodToVariable(MethodDeclaration method, VariableDeclarator var) {
        String methodName = method.getNameAsString();
        String varName = var.getNameAsString();
        try (Session session = driver.session()) {
            session.run(
                    "MATCH (m:Method {name: $methodName}), (v:Variable {name: $varName}) " +
                            "MERGE (m)-[:USES]->(v)",
                    Map.of("methodName", methodName, "varName", varName));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in linkMethodToVariable: " + e.getMessage());
        }
    }

    public void linkMethodToAnnotation(MethodDeclaration method, AnnotationExpr ann) {
        String methodName = method.getNameAsString();
        String annName = ann.getNameAsString();
        if (methodName == null || annName == null) {
            logger.warning(logPrefix + "Cannot link method to annotation: missing name.");
            return;
        }
        try (Session session = driver.session()) {
            session.run(
                    "MERGE (m:Method {name: $methodName}) " +
                            "MERGE (a:Annotation {name: $annName}) " +
                            "MERGE (m)-[:HAS_ANNOTATION]->(a)",
                    Map.of("methodName", methodName, "annName", annName));
            logger.info(logPrefix + "Linked method '" + methodName + "' to annotation '" + annName + "'");
        } catch (Exception e) {
            logger.severe(logPrefix + "Failed to link method '" + methodName + "' to annotation '" + annName + "': "
                    + e.getMessage());
        }
    }

    public void linkClassToAnnotation(ClassOrInterfaceDeclaration clazz, AnnotationExpr ann) {
        String className = clazz.getNameAsString();
        String annName = ann.getNameAsString();
        try (Session session = driver.session()) {
            session.run(
                    "MATCH (c:Class {name: $className}), (a:Annotation {name: $annName}) " +
                            "MERGE (c)-[:HAS_ANNOTATION]->(a)",
                    Map.of("className", className, "annName", annName));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in linkClassToAnnotation: " + e.getMessage());
        }
    }

    public void linkMethodToClass(MethodDeclaration method) {
        String methodName = method.getNameAsString();
        String className = null;

        // Walk up the AST to find the enclosing class or interface
        Optional<Node> current = method.getParentNode();
        while (current.isPresent()) {
            Node node = current.get();
            logger.info(logPrefix + "Checking node: " + node.getClass().getSimpleName());
            if (node instanceof ClassOrInterfaceDeclaration) {
                className = ((ClassOrInterfaceDeclaration) node).getNameAsString();
                break;
            }
            current = node.getParentNode();
        }

        if (className == null) {
            logger.warning("⚠️ Skipping method '" + methodName + "': no enclosing class found.");
            return;
        }

        try (Session session = driver.session()) {
            Map<String, Object> params = new HashMap<>();
            params.put("className", className);
            params.put("methodName", methodName);
            session.run(
                    "MATCH (c:Class {name: $className}), (m:Method {name: $methodName}) " +
                            "MERGE (c)-[:HAS_METHOD]->(m)",
                    params);

            logger.info(logPrefix + "Linked method '" + methodName + "' to class '" + className + "'");
        } catch (Exception e) {
            logger.severe(
                    "❌ Failed to link method '" + methodName + "' to class '" + className + "': " + e.getMessage());
        }
    }

    public void createCallRelation(MethodCallExpr call) {
        try {
            ResolvedMethodDeclaration decl = call.resolve();
            logger.info("Processing declaration: " + decl.getName());
            String caller = call.findAncestor(MethodDeclaration.class)
                    .map(MethodDeclaration::getNameAsString).orElse("unknown");
            String callee = decl.getName();
            try (Session session = driver.session()) {
                session.run("MERGE (caller:Method {name: $caller}) " +
                        "MERGE (callee:Method {name: $callee}) " +
                        "MERGE (caller)-[:CALLS]->(callee)", Map.of("caller", caller, "callee", callee));

                logger.info(logPrefix + "Created call relation from '" + caller + "' to '" + callee + "'");
            } catch (Exception e) {
                logger.severe(logPrefix + "Exception in createCallRelation (inner): " + e.getMessage());
            }
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createCallRelation (resolve): " + e.getMessage());
        }
    }

    /**
     * Scans the database for CALLS and HAS_METHOD relationships and creates the
     * reverse
     * CALLED_BY and BELONGS_TO relationships if missing.
     */
    public void updateReverseRelations() {
        try (Session session = driver.session()) {
            // For each CALLS, create CALLED_BY if missing
            String callsToCalledBy = "MATCH (caller:Method)-[:CALLS]->(callee:Method) " +
                    "MERGE (callee)-[:CALLED_BY]->(caller)";
            session.run(callsToCalledBy);

            // For each HAS_METHOD, create BELONGS_TO if missing
            String hasMethodToBelongsTo = "MATCH (c:Class)-[:HAS_METHOD]->(m:Method) " +
                    "MERGE (m)-[:BELONGS_TO]->(c)";
            session.run(hasMethodToBelongsTo);

            logger.info(logPrefix + "Updated reverse relationships: CALLED_BY and BELONGS_TO.");
        } catch (Exception e) {
            logger.severe(logPrefix + "Failed to update reverse relationships: " + e.getMessage());
        }
    }

    public void createFieldNode(FieldDeclaration field) {
        String name = field.getVariables().get(0).getNameAsString();
        logger.info(logPrefix + "Creating field node: " + name);
        String className = ((ClassOrInterfaceDeclaration) field.getParentNode().get()).getNameAsString();
        logger.info(logPrefix + "Linking field '" + name + "' to class '" + className + "'");

        try (Session session = driver.session()) {
            session.run("MERGE (f:Field {name: $name})", Map.of("name", name));
            session.run("MATCH (c:Class {name: $className}), (f:Field {name: $name}) " +
                    "MERGE (c)-[:HAS_FIELD]->(f)", Map.of("className", className, "name", name));

            logger.info(logPrefix + "Field '" + name + "' linked to class '" + className + "'");
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createFieldNode: " + e.getMessage());
        }
    }

    public void createEnumNode(EnumDeclaration en) {
        String name = en.getNameAsString();
        try (Session session = driver.session()) {
            session.run("MERGE (e:Enum {name: $name})", Map.of("name", name));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createEnumNode: " + e.getMessage());
        }
    }

    public void createConstructorNode(ConstructorDeclaration ctor) {
        String name = ctor.getNameAsString();
        try (Session session = driver.session()) {
            session.run("MERGE (c:Constructor {name: $name})", Map.of("name", name));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createConstructorNode: " + e.getMessage());
        }
    }

    public void createParameterNode(Parameter param) {
        String name = param.getNameAsString();
        try (Session session = driver.session()) {
            session.run("MERGE (p:Parameter {name: $name})", Map.of("name", name));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createParameterNode: " + e.getMessage());
        }
    }

    public void createVariableNode(VariableDeclarator var) {
        String name = var.getNameAsString();
        try (Session session = driver.session()) {
            session.run("MERGE (v:Variable {name: $name})", Map.of("name", name));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createVariableNode: " + e.getMessage());
        }
    }

    public void createPackageNode(PackageDeclaration pkg) {
        String name = pkg.getNameAsString();
        try (Session session = driver.session()) {
            session.run("MERGE (p:Package {name: $name})", Map.of("name", name));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createPackageNode: " + e.getMessage());
        }
    }

    public void createImportNode(ImportDeclaration imp) {
        String name = imp.getNameAsString();
        try (Session session = driver.session()) {
            session.run("MERGE (i:Import {name: $name})", Map.of("name", name));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createImportNode: " + e.getMessage());
        }
    }

    public void createAnnotationNode(AnnotationExpr ann) {
        String name = ann.getNameAsString();
        try (Session session = driver.session()) {
            session.run("MERGE (a:Annotation {name: $name})", Map.of("name", name));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createAnnotationNode: " + e.getMessage());
        }
    }

    public void createTypeNode(TypeParameter type) {
        String name = type.getNameAsString();
        try (Session session = driver.session()) {
            session.run("MERGE (t:Type {name: $name})", Map.of("name", name));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createTypeNode: " + e.getMessage());
        }
    }

    public void createExpressionNode(Expression expr) {
        String value = expr.toString();
        try (Session session = driver.session()) {
            session.run("MERGE (e:Expression {value: $value})", Map.of("value", value));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createExpressionNode: " + e.getMessage());
        }
    }

    public void createLiteralNode(LiteralExpr lit) {
        String value = lit.toString();
        try (Session session = driver.session()) {
            session.run("MERGE (l:Literal {value: $value})", Map.of("value", value));
        } catch (Exception e) {
            logger.severe(logPrefix + "Exception in createLiteralNode: " + e.getMessage());
        }
    }

    public void updateAllNodeDepths() {
        String cypher = "MATCH (n) " +
                "OPTIONAL MATCH p = (n)-[*]->(leaf) " +
                "WHERE NOT (leaf)-->() " +
                "WITH n, collect(length(p)) AS pathLengths " +
                "WITH n, CASE WHEN size(pathLengths) = 0 THEN 0 " +
                "ELSE reduce(maxLen = 0, l IN pathLengths | CASE WHEN l > maxLen THEN l ELSE maxLen END) END AS depth "
                +
                "SET n.depth = depth";
        try (Session session = driver.session()) {
            session.run(cypher);
            logger.info(logPrefix + "Updated depth property for all nodes.");
        } catch (Exception e) {
            logger.severe(logPrefix + "Failed to update node depths: " + e.getMessage());
        }
    }

    public void clearDB() {
        String cypher = "MATCH (n) DETACH DELETE n";
        try (Session session = driver.session()) {
            session.run(cypher);
            logger.info(logPrefix + "Cleared the database.");
        } catch (Exception e) {
            logger.severe(logPrefix + "Failed to clear the database: " + e.getMessage());
        }
    }
}
