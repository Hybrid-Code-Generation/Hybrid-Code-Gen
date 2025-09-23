@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    Collection<Owner> owners = clinicService.findOwnerByLastName(lastName);
    List<OwnerDto> ownerDtos = ownerMapper.toOwnerDtoCollection(owners);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}