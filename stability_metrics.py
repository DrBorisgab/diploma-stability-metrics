def mape_f(res1, res2):
    return (abs(res1 - res2)/(torch.maximum(res1, res2)+1e-8))

def scale(img, scale):
    width, height = img.shape[-2:] #img.size # if error 
    new_height, new_width = int(height * scale), int(width * scale)
    # scaler = torchvision.transforms.Resize(size=(new_height, new_width))
    # resizer = torchvision.transforms.Resize(size=(512, 512))
    # return resizer(scaler(img))
    return  transforms.functional.resize(transforms.functional.resize(img, size=(new_height, new_width)),size=(512, 512))
        
@torch.no_grad()
def get_metrics(augs_dict):
    like_iou = []
    mse = []
    mae = []
    mape = []
    i = 0
    j=0

    for orig_batch in tqdm(orig_dataloader):
        torch.cuda.empty_cache()
        img_orig = orig_batch['img'].cuda()

        img_augs = img_orig.clone().detach()

        for k in range(len(img_orig)):
            for function, parameters in augs_dict.items():
                img_augs[k] = function(img_augs[k], parameters[i])  
                #print('i', i, function, parameters)             
            i+=1

        pred_orig = model(img_orig).sigmoid() 
        pred_augs = model(img_augs).sigmoid() 

        for k in range(len(img_orig)):
            for function, parameters in augs_dict.items():
                if function in [transforms.functional.rotate]:
                    pred_orig[k] = function(pred_orig[k], parameters[j])
            j+=1

        # для устойчивости
        mse_ = torch.mean(torch.mean(mse_f(pred_orig, pred_augs), dim=-1), dim=-1).flatten().cpu()
        mae_ = torch.mean(torch.mean(mae_f(pred_orig, pred_augs), dim=-1), dim=-1).flatten().cpu()
        mape_ = torch.mean(torch.mean(mape_f(pred_orig, pred_augs), dim=-1), dim=-1).flatten().cpu()

        # mse_ = mse_f(pred_orig, pred_augs).flatten().cpu()
        # mae_ = mae_f(pred_orig, pred_augs).flatten().cpu()
        # mape_ = pred_orig, pred_augs.flatten().cpu()


        
        #print(pred_orig, pred_augs)

        pred_orig[pred_orig > 0.501] = 1
        pred_orig[pred_orig != 1] = 0

        pred_augs[pred_augs > 0.501] = 1
        pred_augs[pred_augs != 1] = 0
        iou_ = torch.mean(torch.mean(iou(pred_orig, pred_augs), dim=-1), dim=-1).flatten().cpu()
        #iou_ = iou(pred_orig, pred_augs).flatten().cpu()

        like_iou.append(iou_)
        mse.append(mse_)
        mae.append(mae_)
        mape.append(mape_)

    return like_iou, mse, mae, mape

def loader_metircs_v1(model, orig_dataloader, augs_dict, threshold=0.501):
    torch.cuda.empty_cache()
    gc.collect()
    like_iou, mse, mae, mape = get_metrics(augs_dict=augs_dict)

    print(np.mean(like_iou))

    # для устойчивости
    like_iou = torch.cat(mse).flatten().numpy()
    #print(like_iou.shape)
    mse = torch.cat(mse).flatten().numpy()
    mae = torch.cat(mae).flatten().numpy()
    mape = torch.cat(mape).flatten().numpy()

    print(torch.mean(like_iou))

    return like_iou, mse, mae, mape

    # like_iou = torch.cat(like_iou)
    # mse = torch.cat(mse)
    # mae = torch.cat(mae)
    # mape = torch.cat(mape)
    #return torch.mean(like_iou).item(), torch.mean(mse).item(), torch.mean(mae).item(), torch.mean(mape).item()

def calculate_metrics(model, data_path, augs_dict, augs=None, threshold=0.501):
    img_paths = glob.glob(data_path+'/images/*.tif') # возвращение список (возможно, пустой) путей, соответствующих шаблону 
    mask_paths = glob.glob(data_path+'/masks/*.tif')
    img_paths.sort()
    mask_paths.sort()
    iou = IoU(threshold=threshold)
    scores = {'IoU': [], 'P': [], 'R': [], 'f1': []}
    for i in tqdm(range(len(img_paths))):

        img = Image.open(img_paths[i])
        mask = Image.open(mask_paths[i])

        for function, parameters in augs_dict.items():
            img = function(img, parameters[i])
            if function in [transforms.functional.rotate, scale]:
                #print('struct')
                mask = function(mask, parameters[i])

        img = np.array(img)
        mask = np.array(mask)[:,:,0] # !костыль - хорошо бы все пересохранить

        if augs:
            trans = augs(image=img, mask=mask)
            img, mask = trans['image'], trans['mask']

        mask[mask > 0] = 1
        pred = model(img)

        pred[pred > threshold] = 1
        pred[pred != 1] = 0

        iou_ = iou(pred, mask).item()
    
        scores['IoU'].append(iou_)

        mask, pred = mask.numpy().flatten(), pred.flatten()

        p_ = precision_score(mask, pred)
        r_ = recall_score(mask, pred)
        f1_ = f1_score(mask, pred)
        scores['P'].append(p_)
        scores['R'].append(r_)
        scores['f1'].append(f1_)

        # статистика для каждого изображения
        # score_curr = {'IoU': [], 'P': [], 'R': [], 'f1': []}
        # score_curr['IoU'].append(iou_)
        # score_curr['P'].append(p_)
        # score_curr['R'].append(r_)
        # score_curr['f1'].append(f1_)
 
        #print(pd.DataFrame(score_curr))

    return scores


# superwised metrics
# res_mDErr = orig / diff
# Diff_DErr = (orig - diff) / cnt
def superwised_metrics(model_name,metric_name):
    res_mDErr = []
    Diff_DErr = []
        
    orig = pd.read_csv(f'metrics/files/orig_img_{model_name}.csv')
    distortion = pd.read_csv(f'metrics/files/augs_struct_img_{model_name}.csv')
    
    for i in range(len(orig[metric_name])):
        Err_orig = orig[metric_name][i]
        Err = distortion[metric_name][i]
        #print(Err_orig,Err )

        mDErr = abs(Err - Err_orig) / max(Err_orig, Err)
        DErr = abs(Err_orig - Err)
    
        res_mDErr.append(mDErr)
        Diff_DErr.append(DErr)

    return np.mean(res_mDErr), np.mean(Diff_DErr)



def loader_metircs(model, orig_dataloader, augs_dict, threshold=0.501):
    torch.cuda.empty_cache()
    gc.collect()
    like_iou = []
    mse = []
    mae = []
    mape = []
    i = 0
    j=0
    for orig_batch in tqdm(orig_dataloader): 
        torch.cuda.empty_cache()
        img_orig = orig_batch['img'].cuda()

        img_augs = img_orig.clone().detach()

        for k in range(len(img_orig)):
            for function, parameters in augs_dict.items():
                img_augs[k] = function(img_augs[k], parameters[i])  
                #print('i', i, function, parameters)             
            i+=1

        pred_orig = model(img_orig).sigmoid() 
        pred_augs = model(img_augs).sigmoid() 

        for k in range(len(img_orig)):
            for function, parameters in augs_dict.items():
                if function in [transforms.functional.rotate]:
                    pred_orig[k] = function(pred_orig[k], parameters[j])
            j+=1

        mse_ = mse_f(pred_orig, pred_augs).flatten().cpu()
        mae_ = mae_f(pred_orig, pred_augs).flatten().cpu()
        mape_ = mape_f(pred_orig, pred_augs).flatten().cpu()
        
        #print(pred_orig, pred_augs)

        pred_orig[pred_orig > 0.501] = 1
        pred_orig[pred_orig != 1] = 0

        pred_augs[pred_augs > 0.501] = 1
        pred_augs[pred_augs != 1] = 0

        iou_ = iou(pred_orig, pred_augs).flatten().cpu()

        like_iou.append(iou_)
        mse.append(mse_)
        mae.append(mae_)
        mape.append(mape_)
        #print(mape_)

    like_iou = torch.cat(like_iou)
    mse = torch.cat(mse)
    mae = torch.cat(mae)
    mape = torch.cat(mape)

    #return torch.mean(like_iou).item(), torch.mean(mse).item(), torch.mean(mae).item(), torch.mean(mape).item()
    return like_iou, mse, mae, mape