```java
@Override
public List<Owner> listOwners() {
    String sql = "SELECT id, first_name, last_name, address, city, telephone FROM owners";
    List<Owner> owners = this.jdbcTemplate.query(sql, new OwnerRowMapper());
    loadOwnersPetsAndVisits(owners);
    return owners;
}
```