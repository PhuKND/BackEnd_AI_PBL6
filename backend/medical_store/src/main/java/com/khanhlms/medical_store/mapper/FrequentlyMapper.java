package com.khanhlms.medical_store.mapper;

import com.khanhlms.medical_store.dtos.frequently.request.CreateFrequentlyRequest;
import com.khanhlms.medical_store.dtos.frequently.response.FrequentlyResponse;
import com.khanhlms.medical_store.entities.FrequentlyEntity;
import org.mapstruct.Mapper;

@Mapper(componentModel = "spring")
public abstract class FrequentlyMapper {
    public abstract FrequentlyEntity toEntity(CreateFrequentlyRequest frequentlyRequest);

    public abstract FrequentlyResponse  toResponse(FrequentlyEntity frequentlyEntity);
}
