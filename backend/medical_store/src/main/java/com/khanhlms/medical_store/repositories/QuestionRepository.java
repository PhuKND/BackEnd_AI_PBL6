package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.QuestionEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface QuestionRepository extends JpaRepository<QuestionEntity, String> {

}
