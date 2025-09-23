```java
@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    Collection<Owner> ownerCollection = clinicService.findOwnerByLastName(lastName);
    if (ownerCollection == null || ownerCollection.isEmpty()) {
        return new ResponseEntity<>(HttpStatus.NOT_FOUND);
    }
    List<OwnerDto> ownerDtos = ownerMapper.toOwnerDtoCollection(ownerCollection);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}
```