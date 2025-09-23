```java
@Override
public List<Owner> listOwners() {
    // TODO: implement this method
    final List<Owner> owners = this.namedParameterJdbcTemplate.query(
        "SELECT id, first_name, last_name, address, city, telephone FROM owners ORDER BY id",
        new HashMap<String, Object>(),
        BeanPropertyRowMapper.newInstance(Owner.class)
    );
    loadOwnersPetsAndVisits(owners);
    return owners;
}
```