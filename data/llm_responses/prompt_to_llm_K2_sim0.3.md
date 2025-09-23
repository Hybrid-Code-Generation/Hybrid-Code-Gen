@Override
public ResponseEntity<List<OwnerDto>> listOwners(String lastName) {
    Collection<Owner> owners;
    if (lastName != null && !lastName.isEmpty()) {
        owners = this.clinicService.findOwnersByLastName(lastName);
    } else {
        owners = this.clinicService.findAllOwners();
    }
    List<Owner> ownersList = new ArrayList<>(owners);
    // Load pets and visits for owners
    ((JdbcOwnerRepositoryImpl) this.clinicService.getOwnerRepository()).loadOwnersPetsAndVisits(ownersList);

    List<OwnerDto> ownerDtos = new ArrayList<>();
    for (Owner owner : ownersList) {
        OwnerDto dto = new OwnerDto()
            .firstName(owner.getFirstName())
            .lastName(owner.getLastName())
            .address(owner.getAddress())
            .city(owner.getCity())
            .telephone(owner.getTelephone())
            .id(owner.getId());

        List<PetDto> petDtos = new ArrayList<>();
        if (owner.getPets() != null) {
            for (Pet pet : owner.getPets()) {
                PetDto petDto = new PetDto() // You may need to implement the mapping
                    .id(pet.getId())
                    .name(pet.getName())
                    .birthDate(pet.getBirthDate())
                    .type(pet.getType().getName());
                petDtos.add(petDto);
            }
        }
        dto.setPets(petDtos);
        ownerDtos.add(dto);
    }
    return new ResponseEntity<>(ownerDtos, HttpStatus.OK);
}