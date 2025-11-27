package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.OrderItemEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface OrderItemRepository extends JpaRepository<OrderItemEntity, String> {
}
