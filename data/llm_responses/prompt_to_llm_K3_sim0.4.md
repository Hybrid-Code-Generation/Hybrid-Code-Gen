@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    Collection<Owner> owners;
    if (lastName != null && !lastName.isEmpty()) {
        owners = this.clinicService.findOwnerByLastName(lastName);
    } else {
        owners = this.clinicService.findAllOwners();
    }
    // Assuming loadOwnersPetsAndVisits is available and loads pets and visits for each owner
    loadOwnersPetsAndVisits(new ArrayList<>(owners));
    List<OwnerDto> ownerDtos = ownerMapper.toOwnerDtoCollection(owners);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}