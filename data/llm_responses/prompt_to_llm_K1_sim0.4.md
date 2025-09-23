```java
@Override
public List<Owner> listOwners() {
    // TODO: Complete the method
    List<Owner> owners = new ArrayList<>();
    // Assuming jdbcTemplate is available and mapping is done elsewhere
    owners = this.jdbcTemplate.query(
        "SELECT id, first_name, last_name, address, city, telephone FROM owners",
        (rs, rowNum) -> {
            Owner owner = new Owner();
            owner.setId(rs.getInt("id"));
            owner.setFirstName(rs.getString("first_name"));
            owner.setLastName(rs.getString("last_name"));
            owner.setAddress(rs.getString("address"));
            owner.setCity(rs.getString("city"));
            owner.setTelephone(rs.getString("telephone"));
            return owner;
        }
    );
    loadOwnersPetsAndVisits(owners);
    return owners;
}
```