```java
@Override
public List<Owner> listOwners() {
    // TODO: Complete this method
    List<Owner> owners = this.namedParameterJdbcTemplate.query(
        "SELECT id, first_name, last_name, address, city, telephone FROM owners ORDER BY last_name, first_name",
        new HashMap<String, Object>(),
        BeanPropertyRowMapper.newInstance(Owner.class)
    );
    loadOwnersPetsAndVisits(owners);
    return owners;
}
```