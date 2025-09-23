@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    Collection<Owner> owners = this.clinicService.findOwnerByLastName(lastName);
    if (owners == null || owners.isEmpty()) {
        return new ResponseEntity<>(HttpStatus.NOT_FOUND);
    }
    List<OwnerDto> ownerDtos = this.ownerMapper.toOwnerDtoCollection(owners);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}