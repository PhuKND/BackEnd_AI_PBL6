package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.UserEntity;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<UserEntity, String> {
    Optional<UserEntity> findByUsername(String username);
    Optional<UserEntity> findByEmail(String email);

}
