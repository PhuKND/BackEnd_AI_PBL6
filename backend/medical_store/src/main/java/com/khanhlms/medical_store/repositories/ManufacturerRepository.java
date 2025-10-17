package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.ManufacturerEntity;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ManufacturerRepository extends JpaRepository<ManufacturerEntity, String> {
    Optional<ManufacturerEntity> findByName(String name);
}
