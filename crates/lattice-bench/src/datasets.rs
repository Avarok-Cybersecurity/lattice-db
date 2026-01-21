//! Test dataset generation for benchmarks

use rand::prelude::*;

/// Person data for node creation benchmarks
#[derive(Debug, Clone)]
pub struct PersonData {
    pub name: String,
    pub age: i64,
    pub city: String,
    pub email: String,
}

/// Relationship data for edge creation benchmarks
#[derive(Debug, Clone)]
pub struct RelationshipData {
    pub from_idx: usize,
    pub to_idx: usize,
    pub rel_type: String,
    pub weight: f64,
}

/// Generate a dataset of Person nodes
pub fn generate_people(count: usize, seed: u64) -> Vec<PersonData> {
    let mut rng = StdRng::seed_from_u64(seed);

    let first_names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack",
        "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Ruby", "Sam", "Tina",
    ];
    let last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez",
        "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor",
        "Moore", "Jackson", "Martin",
    ];
    let cities = [
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Phoenix",
        "Philadelphia",
        "San Antonio",
        "San Diego",
        "Dallas",
        "San Jose",
    ];
    let domains = ["gmail.com", "yahoo.com", "outlook.com", "example.com"];

    (0..count)
        .map(|_| {
            let first = first_names[rng.gen_range(0..first_names.len())];
            let last = last_names[rng.gen_range(0..last_names.len())];
            let city = cities[rng.gen_range(0..cities.len())];
            let domain = domains[rng.gen_range(0..domains.len())];
            let age = rng.gen_range(18..80);

            PersonData {
                name: format!("{} {}", first, last),
                age,
                city: city.to_string(),
                email: format!(
                    "{}.{}{}@{}",
                    first.to_lowercase(),
                    last.to_lowercase(),
                    rng.gen_range(1..1000),
                    domain
                ),
            }
        })
        .collect()
}

/// Generate relationships between people (social network style)
pub fn generate_relationships(
    person_count: usize,
    avg_connections: usize,
    seed: u64,
) -> Vec<RelationshipData> {
    let mut rng = StdRng::seed_from_u64(seed);

    let rel_types = ["KNOWS", "FRIENDS_WITH", "WORKS_WITH", "FOLLOWS"];

    let total_relationships = person_count * avg_connections / 2;

    (0..total_relationships)
        .map(|_| {
            let from = rng.gen_range(0..person_count);
            let mut to = rng.gen_range(0..person_count);
            while to == from {
                to = rng.gen_range(0..person_count);
            }

            RelationshipData {
                from_idx: from,
                to_idx: to,
                rel_type: rel_types[rng.gen_range(0..rel_types.len())].to_string(),
                weight: rng.gen_range(0.1..1.0),
            }
        })
        .collect()
}

/// Product data for e-commerce benchmarks
#[derive(Debug, Clone)]
pub struct ProductData {
    pub name: String,
    pub category: String,
    pub price: f64,
    pub stock: i64,
}

/// Generate product catalog
pub fn generate_products(count: usize, seed: u64) -> Vec<ProductData> {
    let mut rng = StdRng::seed_from_u64(seed);

    let categories = [
        "Electronics",
        "Clothing",
        "Books",
        "Home",
        "Sports",
        "Toys",
        "Food",
        "Health",
    ];
    let adjectives = [
        "Premium",
        "Basic",
        "Pro",
        "Ultra",
        "Mini",
        "Mega",
        "Super",
        "Classic",
    ];
    let nouns = [
        "Widget",
        "Gadget",
        "Device",
        "Tool",
        "Item",
        "Product",
        "Thing",
        "Object",
    ];

    (0..count)
        .map(|i| {
            let category = categories[rng.gen_range(0..categories.len())];
            let adj = adjectives[rng.gen_range(0..adjectives.len())];
            let noun = nouns[rng.gen_range(0..nouns.len())];

            ProductData {
                name: format!("{} {} {}", adj, category, noun),
                category: category.to_string(),
                price: (rng.gen_range(1.0f64..1000.0) * 100.0).round() / 100.0,
                stock: rng.gen_range(0..1000),
            }
        })
        .collect()
}

/// Cypher queries used in benchmarks
pub mod queries {
    /// Node creation queries
    pub mod create {
        pub const SINGLE_NODE: &str =
            r#"CREATE (p:Person {name: $name, age: $age, city: $city, email: $email})"#;

        pub const SINGLE_NODE_RETURN: &str =
            r#"CREATE (p:Person {name: $name, age: $age, city: $city, email: $email}) RETURN p"#;
    }

    /// Node lookup queries
    pub mod lookup {
        pub const ALL_NODES: &str = r#"MATCH (n) RETURN n"#;

        pub const BY_LABEL: &str = r#"MATCH (p:Person) RETURN p"#;

        pub const BY_LABEL_LIMIT: &str = r#"MATCH (p:Person) RETURN p LIMIT 10"#;

        pub const BY_PROPERTY: &str = r#"MATCH (p:Person) WHERE p.age > $min_age RETURN p"#;

        pub const BY_PROPERTY_COMPLEX: &str =
            r#"MATCH (p:Person) WHERE p.age > $min_age AND p.city = $city RETURN p"#;

        pub const COUNT_BY_LABEL: &str = r#"MATCH (p:Person) RETURN count(p)"#;
    }

    /// Pattern matching queries
    pub mod pattern {
        pub const SINGLE_HOP: &str =
            r#"MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name LIMIT 100"#;

        pub const TWO_HOP: &str = r#"MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN a.name, c.name LIMIT 100"#;

        pub const FILTERED_PATTERN: &str = r#"MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 AND b.age < 40 RETURN a.name, b.name LIMIT 100"#;
    }

    /// Projection and ordering queries
    pub mod projection {
        pub const SELECT_PROPERTIES: &str =
            r#"MATCH (p:Person) RETURN p.name, p.age, p.city LIMIT 100"#;

        pub const ORDER_BY: &str =
            r#"MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age DESC LIMIT 100"#;

        pub const SKIP_LIMIT: &str = r#"MATCH (p:Person) RETURN p.name SKIP 50 LIMIT 50"#;
    }
}
