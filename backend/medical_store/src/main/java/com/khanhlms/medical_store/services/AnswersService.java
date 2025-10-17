package com.khanhlms.medical_store.services;

import com.khanhlms.medical_store.dtos.answers.request.CreateAnswersRequest;
import com.khanhlms.medical_store.dtos.answers.response.AnswerResponse;
import com.khanhlms.medical_store.entities.AnswersEntity;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.entities.QuestionEntity;
import com.khanhlms.medical_store.entities.UserEntity;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.mapper.AnswersMapper;
import com.khanhlms.medical_store.repositories.AnswersRepository;
import com.khanhlms.medical_store.repositories.QuestionRepository;
import com.khanhlms.medical_store.repositories.UserRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.stereotype.Service;

import java.util.Date;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class AnswersService {
    AnswersRepository answersRepository;
    AnswersMapper  answersMapper;
    UserRepository userRepository;
    QuestionRepository questionRepository;
    public AnswerResponse handCreateAnswer(String productId,String username, String questionID,CreateAnswersRequest request){
        UserEntity user = this.userRepository.findByUsername(username)
                .orElseThrow(()-> new AppException(ErrorCode.USER_NOT_EXISTED));
        QuestionEntity question = this.questionRepository.findById(questionID)
                .orElseThrow(()-> new AppException(ErrorCode.QUESTION_NOT_FOUND));
        ProductsEntity products = question.getProduct();
        if ( ! products.getId().equals(productId) ) {
            throw new AppException(ErrorCode.PRODUCT_NOT_FOUND);
        }
        AnswersEntity answers = this.answersMapper.toAnswersEntity(request);
        answers.setUser(user);
        answers.setQuestion(question);
        answers.setLikesCount(0);
        answers.setCreatedDate(new Date());
        return this.answersMapper.toAnswerResponse(answersRepository.save(answers));
    }
}
