package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.DiscountEntity;
import com.khanhlms.medical_store.entities.ProductsEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface DiscountRepository extends JpaRepository<DiscountEntity, String> {
    Optional<DiscountEntity> findByProduct(ProductsEntity  product);
}
