from services.supabase_service import get_client
 
class PatientService:
 
    @staticmethod
    def create_patient(
        patient_code: str,
        age: int | None,
        sex: str | None,
        notes: str | None,
        researcher_id: str,
    ):
        data = {
            "patient_code": patient_code,
            "age":          age,
            "sex":          sex,
            "notes":        notes,
            "created_by":   researcher_id,
        }
        result = get_client().table("patients").insert(data).execute()
        return result.data
 
    @staticmethod
    def list_patients():
        result = (
            get_client()
            .table("patients")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return result.data
 
    @staticmethod
    def get_patient(patient_id: str):
        result = (
            get_client()
            .table("patients")
            .select("*")
            .eq("id", patient_id)
            .single()
            .execute()
        )
        return result.data
 