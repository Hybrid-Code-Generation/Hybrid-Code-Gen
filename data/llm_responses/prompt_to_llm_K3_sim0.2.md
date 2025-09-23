```java
@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    Collection<Owner> owners;
    if (lastName != null && !lastName.isEmpty()) {
        owners = clinicService.findOwnerByLastName(lastName);
    } else {
        owners = clinicService.findAllOwners();
    }
    // Ensure pets and visits are loaded
    jdbcOwnerRepositoryImpl.loadOwnersPetsAndVisits(new ArrayList<>(owners));
    List<OwnerDto> ownerDtos = ownerMapper.toOwnerDtoCollection(owners);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}
```