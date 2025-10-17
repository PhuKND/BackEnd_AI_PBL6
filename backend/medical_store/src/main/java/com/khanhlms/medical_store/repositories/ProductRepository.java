package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.ProductsEntity;

import com.khanhlms.medical_store.repositories.custom.ProductCustom;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface ProductRepository extends JpaRepository<ProductsEntity, String>, ProductCustom {
    Optional<ProductsEntity> findByName(String name);

    Page<ProductsEntity> findAllByIsActiveTrueAndIsDeletedFalse(Pageable pageable);
}
