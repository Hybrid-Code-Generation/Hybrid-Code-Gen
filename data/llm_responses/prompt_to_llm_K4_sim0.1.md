@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    Collection<Owner> owners = clinicService.findOwnerByLastName(lastName);
    if (owners == null || owners.isEmpty()) {
        return new ResponseEntity<>(HttpStatus.NOT_FOUND);
    }
    // Load pets and visits for each owner (e.g., for JDBC)
    if (owners instanceof List) {
        ownerRepositoryJdbc.loadOwnersPetsAndVisits((List<Owner>) owners);
    } else {
        ownerRepositoryJdbc.loadOwnersPetsAndVisits(new ArrayList<>(owners));
    }
    List<OwnerDto> ownerDtos = ownerMapper.toOwnerDtoCollection(owners);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}