package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.CategoryEntity;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface CategoriesRepository extends JpaRepository<CategoryEntity, String> {
    Optional<CategoryEntity> findByName(String name);
    Page<CategoryEntity> findAll(Pageable pageable);
}
