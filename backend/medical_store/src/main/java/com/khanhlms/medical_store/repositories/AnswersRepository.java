package com.khanhlms.medical_store.repositories;

import com.khanhlms.medical_store.entities.AnswersEntity;
import com.khanhlms.medical_store.entities.QuestionEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface AnswersRepository extends JpaRepository<AnswersEntity, String> {
    Optional<AnswersEntity> findByQuestion(QuestionEntity question);
}
