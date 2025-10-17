package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.FrequentlyEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface FrequentlyRepository extends JpaRepository<FrequentlyEntity, String> {

}
