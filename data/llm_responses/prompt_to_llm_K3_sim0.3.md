@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    Collection<Owner> owners;
    if (lastName != null && !lastName.isEmpty()) {
        owners = clinicService.findOwnerByLastName(lastName);
    } else {
        owners = clinicService.findAllOwners();
    }

    if (owners == null || owners.isEmpty()) {
        return new ResponseEntity<>(Collections.emptyList(), HttpStatus.OK);
    }

    // Optionally load pets and visits if required
    // clinicService.loadOwnersPetsAndVisits(new ArrayList<>(owners));

    List<OwnerDto> ownerDtos = ownerMapper.toOwnerDtoCollection(owners);
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}