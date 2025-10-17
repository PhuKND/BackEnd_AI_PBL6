package com.khanhlms.medical_store.mapper;

import com.khanhlms.medical_store.dtos.answers.response.AnswerResponse;
import com.khanhlms.medical_store.dtos.questions.request.CreateQuestionRequest;
import com.khanhlms.medical_store.dtos.questions.response.QuestionResponse;
import com.khanhlms.medical_store.dtos.requests.CreateUserRequest;
import com.khanhlms.medical_store.entities.AnswersEntity;
import com.khanhlms.medical_store.entities.ProductsEntity;
import com.khanhlms.medical_store.entities.QuestionEntity;
import com.khanhlms.medical_store.entities.UserEntity;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.repositories.ProductRepository;
import com.khanhlms.medical_store.repositories.QuestionRepository;
import com.khanhlms.medical_store.repositories.UserRepository;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;
import org.mapstruct.Named;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.Collections;
import java.util.List;

@Mapper(componentModel = "spring")
public abstract class QuestionMapper {
    @Autowired
    private UserRepository userRepository;
    @Autowired
    private ProductRepository productRepository;
    @Autowired
    private AnswersMapper  answersMapper;
///////////////////////////////////
//    @Mappings({
//        @Mapping(source = "productID", target = "product", qualifiedByName = "mapProductEntity")
//    })
    public abstract QuestionEntity toQuestionEntity(CreateQuestionRequest question);
//    @Named("mapUserEntity")
//    protected UserEntity mapUserEntity(String userID) {
//        if (this.userRepository.findById(userID).isEmpty()) {
//            throw new AppException(ErrorCode.USER_NOT_EXISTED);
//        }
//        return this.userRepository.findById(userID).get();
//    }
//    @Named("mapProductEntity")
//    protected ProductsEntity  mapProductEntity(String productID) {
//        if (this.productRepository.findById(productID).isEmpty()) {
//            throw new AppException(ErrorCode.PRODUCT_NOT_FOUND);
//        }
//        return this.productRepository.findById(productID).get();
//    }
/// ///////////////
    @Mappings({
            @Mapping(source = "user.fullName", target = "userName" ),
            @Mapping(source = "user.avatarUrl", target = "avatarUrl" ),
            @Mapping(source = "id", target = "questionId"),
            @Mapping(source = "answers", target = "answers", qualifiedByName = "mapAnswers")
    })
    public abstract QuestionResponse toQuestionResponse(QuestionEntity questionEntity) ;
    @Named("mapAnswers")
    protected List<AnswerResponse> mapAnswers(List<AnswersEntity> answersEntity) {
        if (answersEntity == null) {
            return Collections.emptyList();
        }
        return answersEntity.stream()
                .map( item -> this.answersMapper.toAnswerResponse(item))
                .toList();
    }
}
