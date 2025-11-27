package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.CartItemEntity;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.entities.UserEntity;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface CartItemsRespository extends JpaRepository<CartItemEntity, String> {
    Optional<CartItemEntity> findById(String id);

    Optional<CartItemEntity> findByProductAndUser(ProductsEntity product, UserEntity user);

    Page<CartItemEntity> findAllByDeletedFalse(Pageable pageable);
}
