package com.khanhlms.medical_store.mapper;

import com.khanhlms.medical_store.dtos.answers.request.CreateAnswersRequest;
import com.khanhlms.medical_store.dtos.answers.response.AnswerResponse;
import com.khanhlms.medical_store.entities.AnswersEntity;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;

@Mapper(componentModel = "spring")
public abstract class AnswersMapper {
    public abstract AnswersEntity toAnswersEntity(CreateAnswersRequest createAnswersRequest);

    @Mappings({
            @Mapping(source = "user.fullName", target = "userName"),
            @Mapping(source = "user.avatarUrl", target = "avatarUrl"),
            @Mapping(source = "id", target = "answerId")
    })
    public abstract AnswerResponse  toAnswerResponse(AnswersEntity answersEntity);
}
