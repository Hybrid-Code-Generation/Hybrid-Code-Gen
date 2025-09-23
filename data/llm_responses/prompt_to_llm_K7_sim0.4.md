@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    List<OwnerDto> ownerDtos;
    Collection<Owner> owners = clinicService.findOwnerByLastName(lastName);
    if (owners == null || owners.isEmpty()) {
        return new ResponseEntity<>(HttpStatus.NOT_FOUND);
    }
    // Load pets and visits for each owner if needed
    // Typically, this would be handled by ClinicService or repository (not shown in provided context)
    // If needed:
    // jdbcOwnerRepositoryImpl.loadOwnersPetsAndVisits(new ArrayList<>(owners));
    ownerDtos = ownerMapper.toOwnerDtoCollection(owners);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}