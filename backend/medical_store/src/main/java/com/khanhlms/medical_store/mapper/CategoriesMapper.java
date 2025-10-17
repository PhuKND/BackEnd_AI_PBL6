package com.khanhlms.medical_store.mapper;

import com.khanhlms.medical_store.dtos.requests.categories.CreateCategoryRequest;
import com.khanhlms.medical_store.dtos.response.categories.CategoryResponse;
import com.khanhlms.medical_store.entities.CategoryEntity;
import com.khanhlms.medical_store.utills.CloudinaryUtils;
import lombok.RequiredArgsConstructor;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;
import org.mapstruct.Named;
import org.mapstruct.factory.Mappers;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.multipart.MultipartFile;

@Mapper(componentModel = "spring")
public abstract class CategoriesMapper {
    @Autowired
    protected  CloudinaryUtils cloudinaryUtils;
    public static CategoriesMapper INSTANCE = Mappers.getMapper(CategoriesMapper.class);

    @Mappings({
            @Mapping(source = "image", target = "thumbnailUrl", qualifiedByName = "mapImage")
    })
    public abstract CategoryEntity toEntity(CreateCategoryRequest categoryRequest);
    public abstract CategoryResponse toResponse(CategoryEntity categoryEntity);

    @Named("mapImage")
    protected String mapImage(MultipartFile multipartFile){
        if(multipartFile == null){
            return null;
        }
        return this.cloudinaryUtils.uploadImageCloddy(multipartFile);
    }
}
