package com.khanhlms.medical_store.mapper;

import com.khanhlms.medical_store.dtos.manufacturer.requests.CreateManufaturerRequest;
import com.khanhlms.medical_store.dtos.manufacturer.response.ManufacturerResponse;
import com.khanhlms.medical_store.entities.ManufacturerEntity;
import com.khanhlms.medical_store.utills.CloudinaryUtils;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;
import org.mapstruct.Named;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.multipart.MultipartFile;

@Mapper(componentModel = "spring")
public abstract class ManufacturerMapper {
    @Autowired
    protected CloudinaryUtils cloudinaryUtils;

    public abstract ManufacturerResponse toResponse(ManufacturerEntity entity) ;

    @Mappings({
            @Mapping(source = "image", target = "thumbnailUrl", qualifiedByName = "mapImage")
    })
    public abstract ManufacturerEntity toEntity(CreateManufaturerRequest request);

    @Named("mapImage")
    protected String mapImage(MultipartFile multipartFile){
        if(multipartFile == null){
            return null;
        }
        return this.cloudinaryUtils.uploadImageCloddy(multipartFile);
    }
}
