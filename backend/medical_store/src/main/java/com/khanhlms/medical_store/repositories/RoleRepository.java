package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.RoleEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface RoleRepository extends JpaRepository<RoleEntity, String> {
    Optional<RoleEntity> findByName(String name);
}
