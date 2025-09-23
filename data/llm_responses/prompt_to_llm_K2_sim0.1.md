@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    Collection<Owner> owners;
    if (lastName != null && !lastName.trim().isEmpty()) {
        owners = clinicService.findOwnerByLastName(lastName);
    } else {
        owners = clinicService.findAllOwners();
    }
    // Load pets and visits for owners
    jdbcOwnerRepositoryImpl.loadOwnersPetsAndVisits(new ArrayList<>(owners));
    // Convert Owners to OwnerDto list
    List<OwnerDto> ownerDtos = ownerMapper.toOwnerDtoCollection(owners);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}