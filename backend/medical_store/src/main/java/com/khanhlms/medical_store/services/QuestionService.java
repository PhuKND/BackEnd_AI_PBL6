package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.dtos.questions.request.CreateQuestionRequest;
import com.khanhlms.medical_store.dtos.questions.response.QuestionResponse;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.entities.QuestionEntity;
import com.khanhlms.medical_store.entities.UserEntity;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.mapper.QuestionMapper;
import com.khanhlms.medical_store.repositories.ProductRepository;
import com.khanhlms.medical_store.repositories.QuestionRepository;
import com.khanhlms.medical_store.repositories.UserRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.stereotype.Repository;
import org.springframework.stereotype.Service;

import java.util.Date;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class QuestionService {
    QuestionRepository  questionRepository;
    QuestionMapper questionMapper;
    ProductRepository productRepository;
    UserRepository userRepository;
    public QuestionResponse handCreateQuestion(String productID, String username, CreateQuestionRequest createQuestionRequest) {
        QuestionEntity questionEntity = this.questionMapper.toQuestionEntity(createQuestionRequest);
        questionEntity.setCreatedDate(new Date());
        UserEntity user = this.userRepository.findByUsername(username)
                .orElseThrow(() ->new AppException(ErrorCode.USER_NOT_EXISTED));
        questionEntity.setUser(user);
        ProductsEntity product = this.productRepository.findById(productID)
                .orElseThrow(() -> new AppException(ErrorCode.PRODUCT_NOT_FOUND));
        questionEntity.setProduct(product);
        return this.questionMapper.toQuestionResponse(this.questionRepository.save(questionEntity)) ;
    }
}
