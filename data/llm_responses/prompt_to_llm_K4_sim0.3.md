@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    List<Owner> owners;
    if (lastName != null && !lastName.isEmpty()) {
        owners = new ArrayList<>(clinicService.findOwnerByLastName(lastName));
    } else {
        owners = new ArrayList<>(clinicService.findAllOwners());
    }

    if (owners.isEmpty()) {
        return new ResponseEntity<>(HttpStatus.NOT_FOUND);
    }

    // Load pets and visits for the owners
    jdbcOwnerRepositoryImpl.loadOwnersPetsAndVisits(owners);

    // Map Owner to OwnerDto
    List<OwnerDto> ownerDtos = ownerMapper.toOwnerDtoCollection(owners);

    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}